The image is the label of a packaged food product.

Extract the following information from the nutrition facts label and return ONLY a JSON object with no additional text:

- calories: total calories per serving (integer)
- serving_size: numeric serving size value (float)
- unit: unit of measurement for serving size (string, e.g., "g", "ml", "oz", "cup")

Use this exact format:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "calories": {
      "type": "integer",
      "description": "Total calories per serving"
    },
    "serving_size": {
      "type": "number",
      "description": "Numeric serving size"
    },
    "unit": {
      "type": "string",
      "description": "Unit of measurement (e.g., g, ml, oz, cups)"
    }
  },
  "required": ["calories", "serving_size", "unit"]
}
```


Example:
```json
{
  "calories": 180,
  "serving_size": 28.0,
  "unit": "g"
}
```

If any value cannot be determined from the label, use null for that field.