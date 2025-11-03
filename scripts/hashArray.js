const crypto = require('crypto');
const fs = require('fs');

// Function to create a human-readable JSON string
function humanReadableStringify(obj) {
    return JSON.stringify(obj, null, 2);
}

// Function to hash a string using SHA-256 with lowercase hex output
function hashString(str) {
    // Convert string to UTF-8 buffer
    const buffer = Buffer.from(str, 'utf8');
    
    // Create hash and get lowercase hex output
    return crypto.createHash('sha256').update(buffer).digest('hex').toLowerCase();
}

// Read and process the JSON file
try {
    const inputData = fs.readFileSync('data/hack_analysis.json', 'utf8');
    const jsonArray = JSON.parse(inputData);
    
    if (!Array.isArray(jsonArray)) {
        console.error('Input must be a JSON array.');
        process.exit(1);
    }

    // Process each item in the array and create output array
    const outputArray = jsonArray.map((item, index) => {
        const humanReadable = humanReadableStringify(item);
        const hash = hashString(humanReadable);
        
        return {
            item_number: index + 1,
            json: humanReadable,
            hash: hash
        };
    });

    // Save output to file
    fs.writeFileSync('data/hack_analysis_hashes.json', JSON.stringify(outputArray, null, 2));
    console.log('Hashes saved to data/hack_analysis_hashes.json');
} catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
} 