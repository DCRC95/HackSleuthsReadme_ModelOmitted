This project connects a **MongoDB database** to an **Ethereum smart contract** on the **Optimism Sepolia** testnet. It automatically listens for specific inserts in the database and logs them immutably on the blockchain via the smart contract.

---

##  Project Structure

- **SmartContracts/** – Contains the `HackSleuth.sol` Solidity contract.
- **scripts/** or root – Node.js backend script to listen for new MongoDB entries and invoke smart contract functions.

---

##  Features

###  Backend Script (Node.js)
- Connects to a MongoDB database.
- Watches the `experiments` collection.
- On insertion of a document matching `hack_analysis_hashes.json`, it:
  - Extracts hash entries from `content`.
  - Sends each valid entry (with title and hash) to the blockchain using the `storeHash()` function on the smart contract.

### 🧾 Smart Contract (`HackSleuth.sol`)
- Stores hash reports with title and timestamp.
- Prevents duplicate hashes.
- Allows:
  - Querying report info by title and transaction index.
  - Ownership transfer by the contract owner.
