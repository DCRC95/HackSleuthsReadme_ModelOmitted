const axios = require('axios');
const dotenv = require('dotenv');
const fs = require('fs');
const path = require('path');
const { createObjectCsvWriter } = require('csv-writer');
dotenv.config();

// Configuration
const ETHERSCAN_API_KEY = process.env.ETHERSCAN_API_KEY;
const ETHERSCAN_API_URL = 'https://api.etherscan.io/api';
const CSV_DIRECTORY = path.join(__dirname, '../data');

// Ensure data directory exists
if (!fs.existsSync(CSV_DIRECTORY)) {
    fs.mkdirSync(CSV_DIRECTORY, { recursive: true });
}

// Initialize CSV writer
const csvWriter = createObjectCsvWriter({
    path: path.join(CSV_DIRECTORY, 'transactions.csv'),
    header: [
        { id: 'blockNumber', title: 'Block Number' },
        { id: 'timestamp', title: 'Timestamp' },
        { id: 'hash', title: 'Transaction Hash' },
        { id: 'from', title: 'From Address' },
        { id: 'to', title: 'To Address' },
        { id: 'value', title: 'Value (ETH)' },
        { id: 'type', title: 'Type' },
        { id: 'token', title: 'Token' }
    ],
    append: true
});

async function getLatestBlockNumber() {
    try {
        const response = await axios.get(ETHERSCAN_API_URL, {
            params: {
                module: 'proxy',
                action: 'eth_blockNumber',
                apikey: ETHERSCAN_API_KEY
            }
        });
        // Subtract 5 blocks to ensure we're processing indexed blocks
        return parseInt(response.data.result, 16) - 5;
    } catch (error) {
        console.error('❌ Error getting latest block:', error.message);
        return null;
    }
}

async function getBlockTransactions(blockNumber) {
    try {
        console.log(`\n📦 Fetching transactions for block ${blockNumber}...`);
        
        // Get block data from Etherscan
        const response = await axios.get(ETHERSCAN_API_URL, {
            params: {
                module: 'block',
                action: 'getblockreward',
                blockno: blockNumber,
                apikey: ETHERSCAN_API_KEY
            }
        });

        if (response.data.status === '0') {
            console.log(`⏳ Block ${blockNumber} not yet indexed, skipping...`);
            return [];
        }

        if (response.data.status === '1') {
            const blockData = response.data.result;
            console.log(`✅ Block ${blockNumber} data retrieved successfully`);
            
            // Get transactions for the block using the correct endpoint
            const txResponse = await axios.get(ETHERSCAN_API_URL, {
                params: {
                    module: 'proxy',
                    action: 'eth_getBlockByNumber',
                    tag: `0x${blockNumber.toString(16)}`,
                    boolean: 'true',
                    apikey: ETHERSCAN_API_KEY
                }
            });

            if (txResponse.data.result) {
                const block = txResponse.data.result;
                const transactions = block.transactions.map(tx => ({
                    blockNumber: blockNumber,
                    timestamp: blockData.timeStamp,
                    hash: tx.hash,
                    from: tx.from,
                    to: tx.to || '', // Some transactions might not have a 'to' address
                    value: (parseInt(tx.value, 16) / 1e18).toString(), // Convert from Wei to ETH
                    type: 'normal',
                    token: ''
                }));

                console.log(`✅ Processed ${transactions.length} transactions`);
                return transactions;
            } else {
                console.log(`⏳ Block ${blockNumber} transactions not yet indexed, skipping...`);
                return [];
            }
        }
        
        return [];
    } catch (error) {
        console.error(`❌ Error fetching block ${blockNumber}:`, error.message);
        if (error.response) {
            console.error('API Error Response:', JSON.stringify(error.response.data, null, 2));
        }
        return [];
    }
}

async function saveTransactionsToCSV(transactions) {
    try {
        console.log('📝 Saving transactions to CSV...');
        await csvWriter.writeRecords(transactions);
        console.log(`✅ Successfully saved ${transactions.length} transactions to CSV`);
    } catch (error) {
        console.error('❌ Error saving to CSV:', error.message);
    }
}

async function processBlock(blockNumber) {
    console.log(`\n🚀 Processing block ${blockNumber}...`);
    
    try {
        const transactions = await getBlockTransactions(blockNumber);
        
        if (transactions.length > 0) {
            await saveTransactionsToCSV(transactions);
            console.log(`📝 Created metadata for block ${blockNumber}`);
            console.log(`📊 Transaction count: ${transactions.length}`);
        }
    } catch (error) {
        console.error(`❌ Error processing block ${blockNumber}:`, error.message);
    }
}

async function startBlockListener() {
    console.log('🚀 Starting block listener...');
    
    try {
        // Start from a block that's definitely indexed (e.g., 100 blocks ago)
        const latestBlock = await getLatestBlockNumber();
        if (!latestBlock) {
            throw new Error('Failed to get latest block number');
        }
        const startBlock = latestBlock - 100; // Start from 100 blocks ago
        console.log(`📦 Starting from block ${startBlock}`);

        // Process the starting block
        await processBlock(startBlock);

        // Set up polling for new blocks
        setInterval(async () => {
            const currentBlock = await getLatestBlockNumber();
            if (currentBlock && currentBlock > startBlock) {
                console.log(`\n🔄 New block detected: ${currentBlock}`);
                await processBlock(currentBlock);
            }
        }, 12000); // Poll every 12 seconds

        console.log('✅ Block listener started successfully');
    } catch (error) {
        console.error('❌ Failed to start block listener:', error.message);
        process.exit(1);
    }
}

// Handle process termination
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down gracefully...');
    process.exit(0);
});

// Start the listener
startBlockListener(); 