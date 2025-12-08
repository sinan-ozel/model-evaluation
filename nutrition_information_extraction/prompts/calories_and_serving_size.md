The image is the label of a packaged food product.

Extract the following information from the nutrition facts label and return ONLY a JSON object with no additional text:

- calories: total calories per serving (integer)
- serving_size: numeric serving size value (float)  
- unit: unit of measurement for serving size (string, e.g., "g", "ml", "oz", "cup")

Use this exact format:

```json
{
  "calories": <integer>,
  "serving_size": <float>,
  "unit": <string>
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