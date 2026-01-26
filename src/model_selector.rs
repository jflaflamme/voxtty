// Model selection system integrated with models.dev
use crate::models_cache::ModelsCache;
use anyhow::{Context, Result};
use dialoguer::{Input, Select};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvider {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub requires_api_key: bool,
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub context_limit: u32,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub input_cost_per_1m: f64,
    pub output_cost_per_1m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider_id: String,
    pub model_id: String,
    pub base_url: String,
    pub api_key: String,
}

pub struct ModelSelector {
    providers: HashMap<String, ModelProvider>,
    cache: ModelsCache,
}

impl ModelSelector {
    pub fn new() -> Self {
        // Load models from cache/API
        let cache = ModelsCache::load();

        // Build providers from cache
        let mut providers = HashMap::new();

        // Add Ollama with discovered local models first (priority)
        let ollama_models: Vec<ModelInfo> = if cache.ollama_models.is_empty() {
            // Fallback models if Ollama not running
            vec![
                ModelInfo {
                    id: "llama3.2".to_string(),
                    name: "Llama 3.2".to_string(),
                    provider: "ollama".to_string(),
                    context_limit: 128000,
                    supports_tools: true,
                    supports_reasoning: false,
                    input_cost_per_1m: 0.0,
                    output_cost_per_1m: 0.0,
                },
                ModelInfo {
                    id: "qwen2.5".to_string(),
                    name: "Qwen 2.5".to_string(),
                    provider: "ollama".to_string(),
                    context_limit: 128000,
                    supports_tools: true,
                    supports_reasoning: false,
                    input_cost_per_1m: 0.0,
                    output_cost_per_1m: 0.0,
                },
            ]
        } else {
            // Use discovered models
            cache
                .ollama_models
                .iter()
                .map(|m| {
                    let name = m.name.split(':').next().unwrap_or(&m.name);
                    ModelInfo {
                        id: m.name.clone(),
                        name: format!("{} ({})", name, format_size(m.size)),
                        provider: "ollama".to_string(),
                        context_limit: 128000, // Default, varies by model
                        supports_tools: true,
                        supports_reasoning: false,
                        input_cost_per_1m: 0.0,
                        output_cost_per_1m: 0.0,
                    }
                })
                .collect()
        };

        providers.insert(
            "ollama".to_string(),
            ModelProvider {
                id: "ollama".to_string(),
                name: "Ollama (Local)".to_string(),
                base_url: "http://localhost:11434/v1".to_string(),
                requires_api_key: false,
                models: ollama_models,
            },
        );

        // Add cloud providers from models.dev cache
        for provider_info in cache.get_llm_providers() {
            let models: Vec<ModelInfo> = cache
                .get_chat_models(&provider_info.id)
                .into_iter()
                .map(|m| ModelInfo {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    provider: provider_info.id.clone(),
                    context_limit: m.limit.context,
                    supports_tools: m.tool_call,
                    supports_reasoning: m.reasoning,
                    input_cost_per_1m: m.cost.input,
                    output_cost_per_1m: m.cost.output,
                })
                .collect();

            if !models.is_empty() {
                providers.insert(
                    provider_info.id.clone(),
                    ModelProvider {
                        id: provider_info.id.clone(),
                        name: provider_info.name.clone(),
                        base_url: cache
                            .get_base_url(&provider_info.id)
                            .unwrap_or_default(),
                        requires_api_key: cache.requires_api_key(&provider_info.id),
                        models,
                    },
                );
            }
        }

        // Fallback: if no providers from cache, use hardcoded defaults
        if providers.len() == 1 {
            // Only Ollama
            Self::add_fallback_providers(&mut providers);
        }

        Self { providers, cache }
    }

    /// Add hardcoded fallback providers when API is unavailable
    fn add_fallback_providers(providers: &mut HashMap<String, ModelProvider>) {
        providers.insert(
            "openai".to_string(),
            ModelProvider {
                id: "openai".to_string(),
                name: "OpenAI".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                requires_api_key: true,
                models: vec![
                    ModelInfo {
                        id: "gpt-4o".to_string(),
                        name: "GPT-4o".to_string(),
                        provider: "openai".to_string(),
                        context_limit: 128000,
                        supports_tools: true,
                        supports_reasoning: false,
                        input_cost_per_1m: 2.50,
                        output_cost_per_1m: 10.00,
                    },
                    ModelInfo {
                        id: "gpt-4o-mini".to_string(),
                        name: "GPT-4o mini".to_string(),
                        provider: "openai".to_string(),
                        context_limit: 128000,
                        supports_tools: true,
                        supports_reasoning: false,
                        input_cost_per_1m: 0.15,
                        output_cost_per_1m: 0.60,
                    },
                ],
            },
        );

        providers.insert(
            "anthropic".to_string(),
            ModelProvider {
                id: "anthropic".to_string(),
                name: "Anthropic".to_string(),
                base_url: "https://api.anthropic.com/v1".to_string(),
                requires_api_key: true,
                models: vec![ModelInfo {
                    id: "claude-sonnet-4-5-20250929".to_string(),
                    name: "Claude Sonnet 4.5".to_string(),
                    provider: "anthropic".to_string(),
                    context_limit: 200000,
                    supports_tools: true,
                    supports_reasoning: true,
                    input_cost_per_1m: 3.00,
                    output_cost_per_1m: 15.00,
                }],
            },
        );
    }

    pub fn interactive_select(&self) -> Result<ModelConfig> {
        println!("\n=== Model Selection ===\n");

        // Sort providers: Ollama first, then alphabetically
        let mut provider_list: Vec<&ModelProvider> = self.providers.values().collect();
        provider_list.sort_by(|a, b| {
            if a.id == "ollama" {
                std::cmp::Ordering::Less
            } else if b.id == "ollama" {
                std::cmp::Ordering::Greater
            } else {
                a.name.cmp(&b.name)
            }
        });

        // Step 1: Select provider
        let provider_names: Vec<String> = provider_list
            .iter()
            .map(|p| {
                format!(
                    "{} ({} models, {})",
                    p.name,
                    p.models.len(),
                    if p.requires_api_key {
                        "API key required"
                    } else {
                        "Local/Free"
                    }
                )
            })
            .collect();

        let provider_idx = Select::new()
            .with_prompt("Select AI provider")
            .items(&provider_names)
            .default(0)
            .interact()
            .context("Failed to select provider")?;

        let provider = provider_list[provider_idx];

        println!("\nProvider: {}", provider.name);
        println!("Base URL: {}", provider.base_url);

        // Step 2: Select model
        let model_names: Vec<String> = provider
            .models
            .iter()
            .map(|m| {
                if m.input_cost_per_1m > 0.0 {
                    format!(
                        "{} ({}k ctx, ${:.2}/${:.2} per 1M)",
                        m.name,
                        m.context_limit / 1000,
                        m.input_cost_per_1m,
                        m.output_cost_per_1m
                    )
                } else {
                    format!("{} ({}k ctx, FREE)", m.name, m.context_limit / 1000)
                }
            })
            .collect();

        let model_idx = Select::new()
            .with_prompt("Select model")
            .items(&model_names)
            .default(0)
            .interact()
            .context("Failed to select model")?;

        let model = &provider.models[model_idx];

        println!("\nModel: {}", model.name);
        println!("ID: {}", model.id);
        println!("Context: {} tokens", model.context_limit);
        if model.input_cost_per_1m > 0.0 {
            println!(
                "Cost: ${:.2} input / ${:.2} output per 1M tokens",
                model.input_cost_per_1m, model.output_cost_per_1m
            );
        } else {
            println!("Cost: FREE (local)");
        }

        // Step 3: Get API key if required
        let api_key = if provider.requires_api_key {
            println!("\nAPI key required for {}", provider.name);

            // Check environment variables first
            let env_key = self
                .cache
                .get_env_var(&provider.id)
                .and_then(|var| std::env::var(&var).ok());

            if let Some(key) = env_key {
                println!("Using API key from environment variable");
                key
            } else {
                Input::<String>::new()
                    .with_prompt("Enter API key")
                    .interact_text()
                    .context("Failed to get API key")?
            }
        } else {
            String::new()
        };

        // Step 4: Confirm base URL (allow override for Ollama)
        let base_url = if provider.id == "ollama" {
            Input::<String>::new()
                .with_prompt("Ollama base URL")
                .default(provider.base_url.clone())
                .interact_text()
                .context("Failed to get base URL")?
        } else {
            provider.base_url.clone()
        };

        Ok(ModelConfig {
            provider_id: provider.id.clone(),
            model_id: model.id.clone(),
            base_url,
            api_key,
        })
    }

    #[allow(dead_code)]
    pub fn get_provider(&self, provider_id: &str) -> Option<&ModelProvider> {
        self.providers.get(provider_id)
    }

    #[allow(dead_code)]
    pub fn list_providers(&self) -> Vec<&ModelProvider> {
        self.providers.values().collect()
    }
}

/// Format byte size to human readable
fn format_size(bytes: u64) -> String {
    if bytes == 0 {
        return "unknown".to_string();
    }
    let gb = bytes as f64 / 1_073_741_824.0;
    if gb >= 1.0 {
        format!("{:.1}GB", gb)
    } else {
        let mb = bytes as f64 / 1_048_576.0;
        format!("{:.0}MB", mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_selector_creation() {
        let selector = ModelSelector::new();
        assert!(!selector.providers.is_empty());
        assert!(selector.get_provider("ollama").is_some());
    }

    #[test]
    fn test_provider_models() {
        let selector = ModelSelector::new();
        let ollama = selector.get_provider("ollama").unwrap();
        assert!(!ollama.requires_api_key);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "unknown");
        assert_eq!(format_size(1_073_741_824), "1.0GB");
        assert_eq!(format_size(4_294_967_296), "4.0GB");
        assert_eq!(format_size(524_288_000), "500MB");
    }
}
