use alloc::{format, string::String};

pub fn generate(schema: &str) -> String {
    format!(
        r#"You must respond with valid JSON that strictly conforms to the following JSON schema:

{schema}

Requirements:
- Your response must be ONLY valid JSON, no additional text, explanations, or markdown
- The JSON must exactly match the schema structure and types
- All required fields must be present
- Use appropriate data types (strings, numbers, booleans, arrays, objects)
- Ensure proper JSON syntax with correct quotes, brackets, and commas
- Do not include any text before or after the JSON

Example format: {{"field1": "value1", "field2": 123}}"#
    )
}
