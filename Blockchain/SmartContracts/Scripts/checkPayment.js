const { ethers } = require("hardhat");

async function main() {
  const [sender] = await ethers.getSigners();
  console.log("Checking payment status for account:", sender.address);

  const contractAddress = "0x44A4e524d55aB9D7c313d3c0cCDaB64c7C7D4AA9";
  
  const OneTimePaywall = await ethers.getContractFactory("OneTimePaywall");
  const paywall = OneTimePaywall.attach(contractAddress);

  // Check payment status
  const hasPaid = await paywall.hasAccess(sender.address);
  
  if (hasPaid) {
    console.log("Payment status: PAID");
    console.log("This address has access to the service");
  } else {
    console.log("Payment status: NOT PAID");
    console.log("This address does not have access to the service");
    
    // Get the required fee for reference
    const fee = await paywall.FEE();
    console.log("Required fee to gain access:", ethers.formatEther(fee), "ETH");
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
}); 