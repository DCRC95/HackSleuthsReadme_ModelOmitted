const { ethers } = require("hardhat");

async function main() {
  const [sender] = await ethers.getSigners();
  console.log("Using account:", sender.address);

  const contractAddress = "0x44A4e524d55aB9D7c313d3c0cCDaB64c7C7D4AA9";
  
  const OneTimePaywall = await ethers.getContractFactory("OneTimePaywall");
  const paywall = OneTimePaywall.attach(contractAddress);

  // Check if already paid
  const hasPaid = await paywall.hasAccess(sender.address);
  if (hasPaid) {
    console.log("You have already paid for the service");
    return;
  }

  // Get the required fee
  const fee = await paywall.FEE();
  console.log("Required fee:", ethers.formatEther(fee), "ETH");

  try {
    // Make the payment
    console.log("Making payment...");
    const tx = await paywall.pay({ value: fee });
    await tx.wait();

    console.log("Payment successful!");
    console.log("Transaction hash:", tx.hash);
  } catch (error) {
    if (error.message.includes("InvalidPaymentAmount")) {
      console.log("Error: Invalid payment amount. Please send exactly", ethers.formatEther(fee), "ETH");
    } else if (error.message.includes("AlreadyPaid")) {
      console.log("Error: You have already paid for the service");
    } else {
      console.error("Error:", error.message);
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
}); 