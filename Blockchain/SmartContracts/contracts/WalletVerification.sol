// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

contract OneTimePaywall is ReentrancyGuard, AccessControl {
    uint256 public constant FEE = 0.0005 ether;
    mapping(address => bool) public hasPaid;

    event PaymentReceived(address indexed payer, uint256 amount);
    event Withdrawal(address indexed to, uint256 amount);
    event BalanceChecked(address indexed checker, uint256 balance);
    event AdminAdded(address indexed admin);
    event AdminRemoved(address indexed admin);
    event PaymentStatusReset(address indexed user, address indexed admin);

    error InvalidPaymentAmount();
    error AlreadyPaid();
    error NoFunds();
    error Unauthorized();

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    function addAdmin(address admin) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        emit AdminAdded(admin);
    }

    function removeAdmin(address admin) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokeRole(DEFAULT_ADMIN_ROLE, admin);
        emit AdminRemoved(admin);
    }

    function pay() external payable nonReentrant {
        if (msg.value != FEE) revert InvalidPaymentAmount();
        if (hasPaid[msg.sender]) revert AlreadyPaid();

        hasPaid[msg.sender] = true;
        emit PaymentReceived(msg.sender, msg.value);
    }

    function withdraw() external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
        uint256 balance = address(this).balance;
        if (balance == 0) revert NoFunds();

        address admin = msg.sender;
        (bool success, ) = payable(admin).call{value: balance}("");
        if (!success) revert("Transfer failed");

        emit Withdrawal(admin, balance);
    }

    function hasAccess(address user) external view returns (bool) {
        return hasPaid[user];
    }

    function checkContractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function resetUserPaymentStatus(address user) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (!hasPaid[user]) revert("User has not paid");
        hasPaid[user] = false;
        emit PaymentStatusReset(user, msg.sender);
    }

    // Prevent direct ETH transfers
    receive() external payable {
        revert("Direct transfers not allowed");
    }

    fallback() external payable {
        revert("Direct transfers not allowed");
    }
}
