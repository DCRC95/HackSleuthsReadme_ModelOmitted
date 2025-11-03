const { ethers } = require("hardhat");

async function main() {
  const [admin] = await ethers.getSigners();
  console.log("Using admin account:", admin.address);

  const contractAddress = "0x44A4e524d55aB9D7c313d3c0cCDaB64c7C7D4AA9";
  
  const OneTimePaywall = await ethers.getContractFactory("OneTimePaywall");
  const paywall = OneTimePaywall.attach(contractAddress);

  // Check if caller is admin
  const isAdmin = await paywall.hasRole(await paywall.DEFAULT_ADMIN_ROLE(), admin.address);
  if (!isAdmin) {
    console.log("Error: Only admins can reset payment status");
    return;
  }

  // Get the user address to reset from command line arguments
  const userAddress = process.argv[2];
  if (!userAddress) {
    console.log("Error: Please provide a user address to reset");
    console.log("Usage: npx hardhat run scripts/resetPaymentStatus.js --network sepolia <user_address>");
    return;
  }

  try {
    console.log(`Resetting payment status for user: ${userAddress}`);
    const tx = await paywall.resetUserPaymentStatus(userAddress);
    await tx.wait();

    console.log("Payment status reset successful!");
    console.log("Transaction hash:", tx.hash);

    // Verify the reset
    const hasPaid = await paywall.hasAccess(userAddress);
    console.log("New payment status:", hasPaid ? "PAID" : "NOT PAID");
  } catch (error) {
    if (error.message.includes("User has not paid")) {
      console.log("Error: User has not paid yet");
    } else {
      console.error("Error:", error.message);
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
}); 