// Dynamic model registry with models.dev API and local caching
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

const MODELS_DEV_API: &str = "https://models.dev/api.json";
const CACHE_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60); // 24 hours
const OLLAMA_API: &str = "http://localhost:11434/api/tags";

/// Model information from models.dev
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub tool_call: bool,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub structured_output: bool,
    #[serde(default)]
    pub open_weights: bool,
    #[serde(default)]
    pub cost: ModelCost,
    #[serde(default)]
    pub limit: ModelLimit,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub reasoning: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelLimit {
    #[serde(default)]
    pub context: u32,
    #[serde(default)]
    pub output: u32,
}

/// Provider information from models.dev
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub api: Option<String>,
    #[serde(default)]
    pub env: Vec<String>,
    #[serde(default)]
    pub models: HashMap<String, ModelInfo>,
}

/// Cached models data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsCache {
    pub providers: HashMap<String, ProviderInfo>,
    pub cached_at: u64,
    pub ollama_models: Vec<OllamaModel>,
}

/// Ollama local model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub size: u64,
    pub modified_at: String,
}

impl ModelsCache {
    /// Get cache file path
    fn cache_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Could not find config directory")?
            .join("voxtty");
        fs::create_dir_all(&config_dir)?;
        Ok(config_dir.join("models_cache.json"))
    }

    /// Load from cache file
    pub fn load_from_cache() -> Result<Option<Self>> {
        let path = Self::cache_path()?;
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)?;
        let cache: Self = serde_json::from_str(&content)?;
        Ok(Some(cache))
    }

    /// Save to cache file
    pub fn save_to_cache(&self) -> Result<()> {
        let path = Self::cache_path()?;
        let content = serde_json::to_string_pretty(self)?;
        fs::write(&path, content)?;
        Ok(())
    }

    /// Check if cache is stale
    pub fn is_stale(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now - self.cached_at > CACHE_MAX_AGE.as_secs()
    }

    /// Fetch fresh data from models.dev API
    pub fn fetch_from_api() -> Result<HashMap<String, ProviderInfo>> {
        let response = reqwest::blocking::get(MODELS_DEV_API)
            .context("Failed to fetch models.dev API")?;

        if !response.status().is_success() {
            anyhow::bail!("models.dev API returned status: {}", response.status());
        }

        let providers: HashMap<String, ProviderInfo> = response.json()?;
        Ok(providers)
    }

    /// Fetch local Ollama models
    pub fn fetch_ollama_models() -> Vec<OllamaModel> {
        #[derive(Deserialize)]
        struct OllamaResponse {
            models: Vec<OllamaModelRaw>,
        }

        #[derive(Deserialize)]
        struct OllamaModelRaw {
            name: String,
            #[serde(default)]
            size: u64,
            #[serde(default)]
            modified_at: String,
        }

        match reqwest::blocking::get(OLLAMA_API) {
            Ok(response) => {
                if let Ok(data) = response.json::<OllamaResponse>() {
                    data.models
                        .into_iter()
                        .map(|m| OllamaModel {
                            name: m.name,
                            size: m.size,
                            modified_at: m.modified_at,
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            }
            Err(_) => Vec::new(), // Ollama not running
        }
    }

    /// Load models - tries cache first, then API, then fallback
    pub fn load() -> Self {
        // Try loading from cache
        if let Ok(Some(cache)) = Self::load_from_cache() {
            if !cache.is_stale() {
                eprintln!("Using cached models data");
                return cache;
            }
            eprintln!("Cache is stale, refreshing...");
        }

        // Try fetching from API
        match Self::fetch_from_api() {
            Ok(providers) => {
                eprintln!("Fetched {} providers from models.dev", providers.len());
                let ollama_models = Self::fetch_ollama_models();
                if !ollama_models.is_empty() {
                    eprintln!("Found {} local Ollama models", ollama_models.len());
                }

                let cache = Self {
                    providers,
                    cached_at: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    ollama_models,
                };

                // Save to cache
                if let Err(e) = cache.save_to_cache() {
                    eprintln!("Failed to save cache: {}", e);
                }

                cache
            }
            Err(e) => {
                eprintln!("Failed to fetch models.dev: {}", e);

                // Try loading stale cache
                if let Ok(Some(cache)) = Self::load_from_cache() {
                    eprintln!("Using stale cache as fallback");
                    return cache;
                }

                // Return empty cache with just Ollama
                eprintln!("Using minimal fallback (Ollama only)");
                Self {
                    providers: HashMap::new(),
                    cached_at: 0,
                    ollama_models: Self::fetch_ollama_models(),
                }
            }
        }
    }

    /// Get providers relevant for voxtty (LLM chat providers)
    pub fn get_llm_providers(&self) -> Vec<&ProviderInfo> {
        // Filter to providers that have chat-capable models
        let relevant_ids = [
            "openai",
            "anthropic",
            "google",
            "deepseek",
            "openrouter",
            "groq",
            "together",
            "fireworks",
            "mistral",
            "cohere",
        ];

        self.providers
            .values()
            .filter(|p| relevant_ids.contains(&p.id.as_str()))
            .collect()
    }

    /// Get models for a specific provider, filtered for chat capability
    pub fn get_chat_models(&self, provider_id: &str) -> Vec<&ModelInfo> {
        self.providers
            .get(provider_id)
            .map(|p| {
                p.models
                    .values()
                    .filter(|m| {
                        // Filter to chat-capable models (has context limit, not embedding/whisper)
                        m.limit.context > 0
                            && !m.family.contains("embed")
                            && !m.family.contains("whisper")
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if API key is required for provider
    pub fn requires_api_key(&self, provider_id: &str) -> bool {
        self.providers
            .get(provider_id)
            .map(|p| !p.env.is_empty())
            .unwrap_or(true)
    }

    /// Get base URL for provider
    pub fn get_base_url(&self, provider_id: &str) -> Option<String> {
        self.providers.get(provider_id).and_then(|p| {
            p.api.clone().or_else(|| {
                // Default URLs for known providers
                match provider_id {
                    "openai" => Some("https://api.openai.com/v1".to_string()),
                    "anthropic" => Some("https://api.anthropic.com/v1".to_string()),
                    "google" => {
                        Some("https://generativelanguage.googleapis.com/v1beta".to_string())
                    }
                    "deepseek" => Some("https://api.deepseek.com/v1".to_string()),
                    "openrouter" => Some("https://openrouter.ai/api/v1".to_string()),
                    "groq" => Some("https://api.groq.com/openai/v1".to_string()),
                    "together" => Some("https://api.together.xyz/v1".to_string()),
                    "fireworks" => Some("https://api.fireworks.ai/inference/v1".to_string()),
                    "mistral" => Some("https://api.mistral.ai/v1".to_string()),
                    "cohere" => Some("https://api.cohere.ai/v1".to_string()),
                    _ => None,
                }
            })
        })
    }

    /// Get environment variable name for provider API key
    pub fn get_env_var(&self, provider_id: &str) -> Option<String> {
        self.providers
            .get(provider_id)
            .and_then(|p| p.env.first().cloned())
            .or_else(|| {
                // Fallback for known providers
                match provider_id {
                    "openai" => Some("OPENAI_API_KEY".to_string()),
                    "anthropic" => Some("ANTHROPIC_API_KEY".to_string()),
                    "google" => Some("GOOGLE_API_KEY".to_string()),
                    "deepseek" => Some("DEEPSEEK_API_KEY".to_string()),
                    "openrouter" => Some("OPENROUTER_API_KEY".to_string()),
                    "groq" => Some("GROQ_API_KEY".to_string()),
                    "together" => Some("TOGETHER_API_KEY".to_string()),
                    "fireworks" => Some("FIREWORKS_API_KEY".to_string()),
                    "mistral" => Some("MISTRAL_API_KEY".to_string()),
                    "cohere" => Some("COHERE_API_KEY".to_string()),
                    _ => None,
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_path() {
        let path = ModelsCache::cache_path();
        assert!(path.is_ok());
        assert!(path.unwrap().ends_with("models_cache.json"));
    }

    #[test]
    fn test_stale_check() {
        let cache = ModelsCache {
            providers: HashMap::new(),
            cached_at: 0, // Very old
            ollama_models: Vec::new(),
        };
        assert!(cache.is_stale());

        let recent_cache = ModelsCache {
            providers: HashMap::new(),
            cached_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ollama_models: Vec::new(),
        };
        assert!(!recent_cache.is_stale());
    }
}
