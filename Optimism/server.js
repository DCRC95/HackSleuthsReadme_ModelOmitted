const { ethers } = require("ethers");
const mongoose = require("mongoose");
require("dotenv").config();
const abi = require("./abi.json");

// Load env variables
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS;
const PRIVATE_KEY = process.env.PRIVATE_KEY;
const MONGO_URI = process.env.MONGO_URI;

// Setup Optimism Sepolia provider
const provider = new ethers.JsonRpcProvider("https://sepolia.optimism.io");
const wallet = new ethers.Wallet(PRIVATE_KEY, provider);
const contract = new ethers.Contract(CONTRACT_ADDRESS, abi, wallet);

// Connect to MongoDB
mongoose.connect(MONGO_URI);
const db = mongoose.connection;
const Experiment = db.collection("experiments");

db.once("open", () => {
  console.log("✅ Connected to MongoDB");

  // Start watching the collection for inserts
  const changeStream = Experiment.watch([{ $match: { operationType: "insert" } }]);
  console.log("👀 Watching for new experiment entries...");

  changeStream.on("change", async (change) => {
    try {
      const newDoc = change.fullDocument;

      if (
        newDoc &&
        newDoc.source_file &&
        /hack_analysis_hashes\.json/i.test(newDoc.source_file) &&
        newDoc.content
      ) {
        const content = newDoc.content;
        const keys = Object.keys(content);

        console.log(`📥 New doc with ${keys.length} entries detected`);

        for (const key of keys) {
          const entry = content[key];
          const title = entry.title || key;
          const hash = entry.hash;
          const timestamp = entry.timestamp || Math.floor(Date.now() / 1000);

          if (!title || !hash) {
            console.warn(`⚠️ Skipping invalid entry: ${key}`);
            continue;
          }

          try {
            const tx = await contract.storeHash(title, hash);
            await tx.wait();
            console.log(`✅ TX stored for ${title}: ${tx.hash}`);
          } catch (err) {
            console.error(`❌ TX Failed for "${title}": ${err.message}`);
          }
        }
      } else {
        console.warn("⚠️ Ignoring document without valid source_file or content");
      }
    } catch (err) {
      console.error("❌ Error handling new insert:", err.message);
    }
  });

  changeStream.on("error", (err) => {
    console.error("❌ Change stream error:", err.message);
  });
});
