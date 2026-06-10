// User "skills" — drop-in markdown files that extend the assistant.
//
// Any `*.md` file placed in ~/.config/voxtty/skills/ is loaded and injected
// into the assistant's system prompt, so the model follows it when a request
// matches. Files are re-read each turn, so dropping a new skill takes effect
// without restarting.
//
// Optional YAML-style frontmatter is supported:
//
//   ---
//   name: weather
//   description: Answer questions about the weather
//   ---
//   When the user asks about weather, call the get_weather MCP tool and speak
//   a one-sentence summary.
//
// Without frontmatter, the file name is used as the skill name and the first
// non-heading line as the description.

use std::fs;
use std::path::PathBuf;

pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
}

/// `~/.config/voxtty/skills`
fn skills_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("voxtty").join("skills"))
}

/// Load all skill markdown files, sorted by name. Missing dir → empty.
pub fn load_skills() -> Vec<Skill> {
    let Some(dir) = skills_dir() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("skill")
            .to_string();
        skills.push(parse_skill(&content, stem));
    }
    skills.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    skills
}

fn parse_skill(content: &str, fallback_name: String) -> Skill {
    let mut name = fallback_name;
    let mut description = String::new();
    let body: String;

    // Optional `---` frontmatter block at the very start.
    if let Some(rest) = content.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            let frontmatter = &rest[..end];
            body = rest[end + "\n---".len()..].trim_start().to_string();
            for line in frontmatter.lines() {
                if let Some(v) = line.strip_prefix("name:") {
                    name = v.trim().trim_matches('"').to_string();
                } else if let Some(v) = line.strip_prefix("description:") {
                    description = v.trim().trim_matches('"').to_string();
                }
            }
        } else {
            body = content.to_string();
        }
    } else {
        body = content.to_string();
    }

    if description.is_empty() {
        description = body
            .lines()
            .find(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .unwrap_or("")
            .trim()
            .to_string();
    }

    Skill {
        name,
        description,
        body,
    }
}

/// Render loaded skills as a system-prompt section. Empty string if none exist.
pub fn skills_prompt_section() -> String {
    let skills = load_skills();
    if skills.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "\n\n## USER SKILLS\n\n\
         The user has provided the following skills. When a request matches one, \
         follow its instructions.\n",
    );
    for skill in &skills {
        out.push_str(&format!("\n### {}\n", skill.name));
        if !skill.description.is_empty() {
            out.push_str(&format!("_{}_\n\n", skill.description));
        }
        out.push_str(skill.body.trim());
        out.push('\n');
    }
    out
}
