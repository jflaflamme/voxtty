// Model selection system integrated with models.dev
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
}

impl ModelSelector {
    pub fn new() -> Self {
        let mut providers = HashMap::new();

        // OpenAI
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
                    ModelInfo {
                        id: "o1".to_string(),
                        name: "o1 (Reasoning)".to_string(),
                        provider: "openai".to_string(),
                        context_limit: 200000,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 15.00,
                        output_cost_per_1m: 60.00,
                    },
                    ModelInfo {
                        id: "o1-mini".to_string(),
                        name: "o1-mini (Reasoning)".to_string(),
                        provider: "openai".to_string(),
                        context_limit: 128000,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 1.10,
                        output_cost_per_1m: 4.40,
                    },
                ],
            },
        );

        // Anthropic
        providers.insert(
            "anthropic".to_string(),
            ModelProvider {
                id: "anthropic".to_string(),
                name: "Anthropic".to_string(),
                base_url: "https://api.anthropic.com/v1".to_string(),
                requires_api_key: true,
                models: vec![
                    ModelInfo {
                        id: "claude-sonnet-4-5-20250929".to_string(),
                        name: "Claude Sonnet 4.5".to_string(),
                        provider: "anthropic".to_string(),
                        context_limit: 200000,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 3.00,
                        output_cost_per_1m: 15.00,
                    },
                    ModelInfo {
                        id: "claude-3-5-haiku-20241022".to_string(),
                        name: "Claude Haiku 3.5".to_string(),
                        provider: "anthropic".to_string(),
                        context_limit: 200000,
                        supports_tools: true,
                        supports_reasoning: false,
                        input_cost_per_1m: 0.80,
                        output_cost_per_1m: 4.00,
                    },
                ],
            },
        );

        // Google
        providers.insert(
            "google".to_string(),
            ModelProvider {
                id: "google".to_string(),
                name: "Google".to_string(),
                base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
                requires_api_key: true,
                models: vec![
                    ModelInfo {
                        id: "gemini-2.5-flash".to_string(),
                        name: "Gemini 2.5 Flash".to_string(),
                        provider: "google".to_string(),
                        context_limit: 1048576,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 0.30,
                        output_cost_per_1m: 2.50,
                    },
                    ModelInfo {
                        id: "gemini-2.5-pro".to_string(),
                        name: "Gemini 2.5 Pro".to_string(),
                        provider: "google".to_string(),
                        context_limit: 1048576,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 1.25,
                        output_cost_per_1m: 10.00,
                    },
                ],
            },
        );

        // DeepSeek
        providers.insert(
            "deepseek".to_string(),
            ModelProvider {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                requires_api_key: true,
                models: vec![
                    ModelInfo {
                        id: "deepseek-chat".to_string(),
                        name: "DeepSeek Chat".to_string(),
                        provider: "deepseek".to_string(),
                        context_limit: 128000,
                        supports_tools: true,
                        supports_reasoning: false,
                        input_cost_per_1m: 0.57,
                        output_cost_per_1m: 1.68,
                    },
                    ModelInfo {
                        id: "deepseek-reasoner".to_string(),
                        name: "DeepSeek Reasoner (R1)".to_string(),
                        provider: "deepseek".to_string(),
                        context_limit: 128000,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 0.57,
                        output_cost_per_1m: 1.68,
                    },
                ],
            },
        );

        // Ollama (Local)
        providers.insert(
            "ollama".to_string(),
            ModelProvider {
                id: "ollama".to_string(),
                name: "Ollama (Local)".to_string(),
                base_url: "http://localhost:11434/v1".to_string(),
                requires_api_key: false,
                models: vec![
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
                        id: "mistral".to_string(),
                        name: "Mistral".to_string(),
                        provider: "ollama".to_string(),
                        context_limit: 32000,
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
                ],
            },
        );

        // OpenRouter (Aggregator)
        providers.insert(
            "openrouter".to_string(),
            ModelProvider {
                id: "openrouter".to_string(),
                name: "OpenRouter".to_string(),
                base_url: "https://openrouter.ai/api/v1".to_string(),
                requires_api_key: true,
                models: vec![
                    ModelInfo {
                        id: "anthropic/claude-sonnet-4.5".to_string(),
                        name: "Claude Sonnet 4.5 (via OpenRouter)".to_string(),
                        provider: "openrouter".to_string(),
                        context_limit: 1000000,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 3.00,
                        output_cost_per_1m: 15.00,
                    },
                    ModelInfo {
                        id: "openai/gpt-4o".to_string(),
                        name: "GPT-4o (via OpenRouter)".to_string(),
                        provider: "openrouter".to_string(),
                        context_limit: 128000,
                        supports_tools: true,
                        supports_reasoning: false,
                        input_cost_per_1m: 2.50,
                        output_cost_per_1m: 10.00,
                    },
                    ModelInfo {
                        id: "google/gemini-2.5-flash".to_string(),
                        name: "Gemini 2.5 Flash (via OpenRouter)".to_string(),
                        provider: "openrouter".to_string(),
                        context_limit: 1048576,
                        supports_tools: true,
                        supports_reasoning: true,
                        input_cost_per_1m: 0.30,
                        output_cost_per_1m: 2.50,
                    },
                ],
            },
        );

        Self { providers }
    }

    pub fn interactive_select(&self) -> Result<ModelConfig> {
        println!("\n=== Model Selection ===\n");

        // Step 1: Select provider
        let provider_names: Vec<String> = self
            .providers
            .values()
            .map(|p| {
                format!(
                    "{} ({})",
                    p.name,
                    if p.requires_api_key {
                        "API key required"
                    } else {
                        "Local/Free"
                    }
                )
            })
            .collect();

        let provider_ids: Vec<String> = self.providers.keys().cloned().collect();

        let provider_idx = Select::new()
            .with_prompt("Select AI provider")
            .items(&provider_names)
            .default(0)
            .interact()
            .context("Failed to select provider")?;

        let provider_id = &provider_ids[provider_idx];
        let provider = self
            .providers
            .get(provider_id)
            .context("Provider not found")?;

        println!("\nProvider: {}", provider.name);
        println!("Base URL: {}", provider.base_url);

        // Step 2: Select model
        let model_names: Vec<String> = provider
            .models
            .iter()
            .map(|m| {
                format!(
                    "{} ({}k ctx, ${:.2}/${:.2} per 1M tokens)",
                    m.name,
                    m.context_limit / 1000,
                    m.input_cost_per_1m,
                    m.output_cost_per_1m
                )
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
        println!(
            "Cost: ${:.2} input / ${:.2} output per 1M tokens",
            model.input_cost_per_1m, model.output_cost_per_1m
        );

        // Step 3: Get API key if required
        let api_key = if provider.requires_api_key {
            println!("\nAPI key required for {}", provider.name);

            // Check environment variables first
            let env_key = match provider_id.as_str() {
                "openai" => std::env::var("OPENAI_API_KEY").ok(),
                "anthropic" => std::env::var("ANTHROPIC_API_KEY").ok(),
                "google" => std::env::var("GOOGLE_API_KEY").ok(),
                "deepseek" => std::env::var("DEEPSEEK_API_KEY").ok(),
                "openrouter" => std::env::var("OPENROUTER_API_KEY").ok(),
                _ => None,
            };

            if let Some(key) = env_key {
                println!("✓ Using API key from environment variable");
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

        // Step 4: Confirm base URL (allow override)
        let base_url = if provider_id == "ollama" {
            Input::<String>::new()
                .with_prompt("Ollama base URL")
                .default(provider.base_url.clone())
                .interact_text()
                .context("Failed to get base URL")?
        } else {
            provider.base_url.clone()
        };

        Ok(ModelConfig {
            provider_id: provider_id.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_selector_creation() {
        let selector = ModelSelector::new();
        assert!(!selector.providers.is_empty());
        assert!(selector.get_provider("openai").is_some());
        assert!(selector.get_provider("ollama").is_some());
    }

    #[test]
    fn test_provider_models() {
        let selector = ModelSelector::new();
        let openai = selector.get_provider("openai").unwrap();
        assert!(!openai.models.is_empty());
        assert!(openai.requires_api_key);

        let ollama = selector.get_provider("ollama").unwrap();
        assert!(!ollama.requires_api_key);
    }
}
