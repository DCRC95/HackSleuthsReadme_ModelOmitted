const { ethers } = require("hardhat");

async function main() {
  const [signer] = await ethers.getSigners();
  console.log("Checking contract balance as:", signer.address);

  const contractAddress = "0x44A4e524d55aB9D7c313d3c0cCDaB64c7C7D4AA9";
  
  const OneTimePaywall = await ethers.getContractFactory("OneTimePaywall");
  const paywall = OneTimePaywall.attach(contractAddress);

  // Check if caller is admin
  const isAdmin = await paywall.hasRole(await paywall.DEFAULT_ADMIN_ROLE(), signer.address);
  if (!isAdmin) {
    console.log("Error: Only admins can check the balance");
    return;
  }

  // Get the contract balance
  const balance = await ethers.provider.getBalance(contractAddress);
  console.log("Contract balance:", ethers.formatEther(balance), "ETH");

  // Call the checkContractBalance function
  console.log("Checking balance through contract function...");
  const contractBalance = await paywall.checkContractBalance();
  console.log("Contract balance from function:", ethers.formatEther(contractBalance), "ETH");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
}); 