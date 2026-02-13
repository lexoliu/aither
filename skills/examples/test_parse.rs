#![allow(missing_docs)]

use aither_skills::{Skill, SkillLoader};
use futures_lite::future::block_on;

fn main() {
    // Test parsing the skill file directly
    let content = std::fs::read_to_string("./outputs/slide/SKILL.md").unwrap();
    println!(
        "File content first 200 chars:\n{}\n---",
        &content[..200.min(content.len())]
    );

    match Skill::parse(&content) {
        Ok(skill) => println!(
            "Parsed skill: {} - {}",
            skill.name,
            &skill.description[..50.min(skill.description.len())]
        ),
        Err(e) => println!("Parse error: {e:?}"),
    }

    // Test loader
    println!("\nTesting loader...");
    let loader = SkillLoader::new().add_path("./outputs");
    match block_on(loader.load_all()) {
        Ok(skills) => {
            println!("Loaded {} skills", skills.len());
            for skill in skills {
                println!(
                    "  - {}: {}",
                    skill.name,
                    &skill.description[..50.min(skill.description.len())]
                );
            }
        }
        Err(e) => println!("Loader error: {e:?}"),
    }
}
