# OneTimePaywall Smart Contract

## Contract Overview
The OneTimePaywall contract is a Solidity smart contract that implements a one-time payment system for access control. Users can pay a fixed fee to gain permanent access to a service or content.

**Contract Address:** [0x44A4e524d55aB9D7c313d3c0cCDaB64c7C7D4AA9]
(https://sepolia.etherscan.io/address/0x44A4e524d55aB9D7c313d3c0cCDaB64c7C7D4AA9#code)

## Contract Features

### Core Functionality
- One-time payment system with fixed fee (0.0005 ETH)
- Role-based access control for administrators
- Payment status tracking
- Admin fund withdrawal capabilities

### Security Features
- AccessControl for admin management
- Prevention of direct ETH transfers
- Custom error handling
- Non-reentrant modifiers on critical functions

### Events
- `PaymentReceived`: Emitted when a user successfully pays
- `Withdrawal`: Emitted when an admin withdraws funds
- `AdminAdded`: Emitted when a new admin is added
- `AdminRemoved`: Emitted when an admin is removed
- `PaymentStatusReset`: Emitted when an admin resets a user's payment status

## Scripts

### 1. Payment Scripts

#### payForService.js
Makes a payment to gain access to the service.
```bash
npx hardhat run scripts/payForService.js --network sepolia
```
- Checks if user has already paid
- Sends exact fee amount (0.0005 ETH)
- Handles payment errors gracefully

#### checkPayment.js
Checks the payment status of an address.
```bash
npx hardhat run scripts/checkPayment.js --network sepolia
```
- Returns current payment status
- Shows required fee if not paid

### 2. Admin Scripts

#### checkContractBalance.js
Allows admins to check the contract's ETH balance.
```bash
npx hardhat run scripts/checkContractBalance.js --network sepolia
```
- Requires admin role
- Shows both raw balance and formatted ETH amount

#### resetPaymentStatus.js
Allows admins to reset a user's payment status.
```bash
npx hardhat run scripts/resetPaymentStatus.js --network sepolia <user_address>
```
- Requires admin role
- Takes user address as parameter
- Verifies reset was successful

### 3. Deployment Scripts

#### deployOneTimePaywall.js
Deploys the contract to the network.
```bash
npx hardhat run scripts/deployOneTimePaywall.js --network sepolia
```
- Sets deployer as initial admin
- Verifies admin role assignment

#### verifyOneTimePaywall.js
Verifies the contract on Etherscan.
```bash
npx hardhat run scripts/verifyOneTimePaywall.js --network sepolia
```
- Verifies contract source code
- Makes contract code publicly viewable

## Contract Functions

### User Functions
- `pay()`: Make a one-time payment
- `hasAccess(address)`: Check if an address has paid

### Admin Functions
- `addAdmin(address)`: Add a new admin
- `removeAdmin(address)`: Remove an admin
- `withdraw()`: Withdraw contract funds
- `checkContractBalance()`: View contract balance
- `resetUserPaymentStatus(address)`: Reset user's payment status

## Error Handling
- `InvalidPaymentAmount`: Wrong payment amount
- `AlreadyPaid`: User already has access
- `NoFunds`: No funds to withdraw
- `Unauthorized`: Access control violation

## Future Enhancements
1. Chainlink Price Feed Integration
   - Maintain consistent USD pricing
   - Automatic fee adjustments
   - Price feed staleness checks

2. Additional Features
   - Pause mechanism
   - Configurable fees
   - Batch operations
   - Maximum admin limit
   - Time-based access control


## Development
- Solidity version: ^0.8.19
- OpenZeppelin contracts
- Hardhat development environment
- Sepolia testnet deployment

