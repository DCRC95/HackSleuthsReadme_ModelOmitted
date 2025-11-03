// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract HackSleuth {
    struct Report {
        string title;
        string hash;
        uint256 timestamp;
    }

    address public owner;

    mapping(string => bool) private hashExists;
    mapping(string => string[]) private titleToHashes;
    mapping(string => Report) private reportData;
    mapping(string => bool) private titleExists;

    event HashStored(string title, string hash, uint256 timestamp);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    // ✅ Store a unique hash under a title
    function storeHash(string memory title, string memory hash) external {
        require(!hashExists[hash], "Hash already stored");

        reportData[hash] = Report({
            title: title,
            hash: hash,
            timestamp: block.timestamp
        });

        titleToHashes[title].push(hash);
        hashExists[hash] = true;
        titleExists[title] = true;

        emit HashStored(title, hash, block.timestamp);
    }

    // ✅ Combine: check title + number of transactions
    function getTitleInfo(string memory title)
        external
        view
        returns (bool exists, uint256 transactionCount)
    {
        exists = titleExists[title];
        transactionCount = titleToHashes[title].length;
    }

    // ✅ Get report by human-friendly transaction number (1 = oldest, N = latest)
    function getReportByTransactionNumber(string memory title, uint256 txNumber)
        external
        view
        returns (
            uint256 transactionNumber,
            string memory hash,
            uint256 timestamp
        )
    {
        uint256 count = titleToHashes[title].length;
        require(txNumber > 0 && txNumber <= count, "Transaction number out of range");

        string memory hashValue = titleToHashes[title][txNumber - 1];
        Report memory r = reportData[hashValue];

        return (txNumber, r.hash, r.timestamp);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid new owner");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}
