// MCP (Model Context Protocol) tool integration
// Spawns MCP servers as child processes and communicates via JSON-RPC over stdio

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

/// Configuration for a single MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Top-level config file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub servers: Vec<McpServerConfig>,
}

/// Tool definition from an MCP server
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

impl McpToolDef {
    /// Convert to OpenAI-compatible tool JSON
    pub fn to_openai_tool(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        })
    }
}

/// Client for a single MCP server process
pub struct McpClient {
    name: String,
    child: Child,
    request_id: Mutex<u64>,
    tools: Vec<McpToolDef>,
}

impl McpClient {
    /// Spawn and initialize an MCP server
    pub fn connect(config: &McpServerConfig) -> Result<Self> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server '{}'", config.name))?;

        let mut client = Self {
            name: config.name.clone(),
            child,
            request_id: Mutex::new(1),
            tools: Vec::new(),
        };

        // Send initialize request
        client.initialize()?;

        // Discover tools
        client.tools = client.discover_tools()?;

        Ok(client)
    }

    fn next_id(&self) -> u64 {
        let mut id = self.request_id.lock().unwrap();
        let current = *id;
        *id += 1;
        current
    }

    fn send_request(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let id = self.next_id();
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        let stdin = self
            .child
            .stdin
            .as_mut()
            .context("MCP server stdin not available")?;

        let request_str = serde_json::to_string(&request)?;
        writeln!(stdin, "{}", request_str)?;
        stdin.flush()?;

        // Read response with timeout
        let stdout = self
            .child
            .stdout
            .as_mut()
            .context("MCP server stdout not available")?;

        let mut reader = BufReader::new(stdout);
        let mut line = String::new();

        // Simple blocking read - the MCP server should respond quickly
        // We set a 30s timeout via the child process
        reader
            .read_line(&mut line)
            .with_context(|| format!("Failed to read response from MCP server '{}'", self.name))?;

        if line.is_empty() {
            anyhow::bail!("MCP server '{}' closed connection", self.name);
        }

        let response: serde_json::Value = serde_json::from_str(line.trim())
            .with_context(|| format!("Invalid JSON from MCP server '{}': {}", self.name, line.trim()))?;

        // Check for JSON-RPC error
        if let Some(error) = response.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("MCP server '{}' error: {}", self.name, msg);
        }

        Ok(response.get("result").cloned().unwrap_or(serde_json::Value::Null))
    }

    fn initialize(&mut self) -> Result<()> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "voxtty",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let result = self.send_request("initialize", params)?;

        // Send initialized notification
        let stdin = self
            .child
            .stdin
            .as_mut()
            .context("MCP server stdin not available")?;

        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });

        writeln!(stdin, "{}", serde_json::to_string(&notification)?)?;
        stdin.flush()?;

        if let Some(info) = result.get("serverInfo") {
            let name = info.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
            let version = info.get("version").and_then(|v| v.as_str()).unwrap_or("?");
            eprintln!("  MCP server '{}' initialized: {} v{}", self.name, name, version);
        }

        Ok(())
    }

    fn discover_tools(&mut self) -> Result<Vec<McpToolDef>> {
        let result = self.send_request("tools/list", serde_json::json!({}))?;

        let tools_array = result
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();

        let mut tools = Vec::new();
        for tool in tools_array {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(serde_json::json!({"type": "object", "properties": {}}));

            if !name.is_empty() {
                tools.push(McpToolDef {
                    name,
                    description,
                    input_schema,
                });
            }
        }

        Ok(tools)
    }

    /// Call a tool on this MCP server
    pub fn call_tool(&mut self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.send_request("tools/call", params)?;

        // Extract text content from result
        if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            let texts: Vec<&str> = content
                .iter()
                .filter_map(|item| {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        item.get("text").and_then(|t| t.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(texts.join("\n"))
        } else {
            // Return raw result as string
            Ok(serde_json::to_string(&result)?)
        }
    }

    /// Get tools discovered from this server
    pub fn tools(&self) -> &[McpToolDef] {
        &self.tools
    }

    /// Shutdown the MCP server
    pub fn shutdown(&mut self) {
        // Try to kill the child process gracefully
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Manages multiple MCP server connections
pub struct McpManager {
    clients: Vec<McpClient>,
    /// Map from tool name to index in clients vec
    tool_routing: HashMap<String, usize>,
}

impl McpManager {
    /// Load config and connect to all MCP servers
    pub fn from_config(config: &McpConfig) -> Self {
        let mut clients = Vec::new();
        let mut tool_routing = HashMap::new();

        for server_config in &config.servers {
            match McpClient::connect(server_config) {
                Ok(client) => {
                    let client_idx = clients.len();
                    // Register tool routing
                    for tool in client.tools() {
                        tool_routing.insert(tool.name.clone(), client_idx);
                    }
                    clients.push(client);
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: Failed to connect to MCP server '{}': {}",
                        server_config.name, e
                    );
                }
            }
        }

        Self {
            clients,
            tool_routing,
        }
    }

    /// Load config from the default path, trying multiple formats:
    /// 1. ~/.config/voxtty/mcp_servers.toml (voxtty native TOML format)
    /// 2. .mcp.json in current directory (Claude Code format)
    pub fn load_config() -> Result<McpConfig> {
        // Try voxtty native TOML format first
        let config_path = dirs::config_dir()
            .context("No config directory found")?
            .join("voxtty")
            .join("mcp_servers.toml");

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read {}", config_path.display()))?;
            let config: McpConfig = toml::from_str(&content)
                .with_context(|| format!("Failed to parse {}", config_path.display()))?;
            return Ok(config);
        }

        // Try Claude Code .mcp.json format in current directory
        let claude_path = std::path::Path::new(".mcp.json");
        if claude_path.exists() {
            return Self::load_claude_code_config(claude_path);
        }

        anyhow::bail!(
            "No MCP config found. Create ~/.config/voxtty/mcp_servers.toml or .mcp.json"
        )
    }

    /// Parse Claude Code .mcp.json format into voxtty McpConfig
    fn load_claude_code_config(path: &std::path::Path) -> Result<McpConfig> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse {}", path.display()))?;

        let servers_obj = json
            .get("mcpServers")
            .and_then(|s| s.as_object())
            .context("No 'mcpServers' key in .mcp.json")?;

        let mut servers = Vec::new();
        for (name, config) in servers_obj {
            // Skip non-stdio servers
            let server_type = config
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("stdio");
            if server_type != "stdio" {
                eprintln!(
                    "  Skipping MCP server '{}': unsupported type '{}'",
                    name, server_type
                );
                continue;
            }

            let command = config
                .get("command")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();

            if command.is_empty() {
                continue;
            }

            let args: Vec<String> = config
                .get("args")
                .and_then(|a| a.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let env: HashMap<String, String> = config
                .get("env")
                .and_then(|e| e.as_object())
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default();

            servers.push(McpServerConfig {
                name: name.clone(),
                command,
                args,
                env,
            });
        }

        eprintln!("  Loaded {} server(s) from .mcp.json", servers.len());
        Ok(McpConfig { servers })
    }

    /// Get all tools from all connected servers
    pub fn all_tools(&self) -> Vec<&McpToolDef> {
        self.clients.iter().flat_map(|c| c.tools()).collect()
    }

    /// Convert all tools to OpenAI-compatible format
    pub fn to_openai_tools(&self) -> Vec<serde_json::Value> {
        self.all_tools().iter().map(|t| t.to_openai_tool()).collect()
    }

    /// Call a tool by name, routing to the correct server
    pub fn call_tool(&mut self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let client_idx = self
            .tool_routing
            .get(name)
            .copied()
            .with_context(|| format!("Unknown MCP tool: {}", name))?;

        self.clients[client_idx].call_tool(name, arguments)
    }

    /// Number of connected servers
    pub fn server_count(&self) -> usize {
        self.clients.len()
    }

    /// Total number of tools across all servers
    pub fn tool_count(&self) -> usize {
        self.tool_routing.len()
    }

    /// Shutdown all MCP servers
    pub fn shutdown(&mut self) {
        for client in &mut self.clients {
            client.shutdown();
        }
    }
}

impl Drop for McpManager {
    fn drop(&mut self) {
        self.shutdown();
    }
}
