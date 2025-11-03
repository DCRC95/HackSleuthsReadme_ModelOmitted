const { ethers } = require("hardhat");

async function main() {
  // Get the contract address from the deployment
  const contractAddress = "YOUR_DEPLOYED_CONTRACT_ADDRESS";
  
  // Get the signers
  const [admin, newAdmin] = await ethers.getSigners();
  console.log("Current admin address:", admin.address);
  console.log("New admin address:", newAdmin.address);

  // Get the contract instance
  const OneTimePaywall = await ethers.getContractFactory("OneTimePaywall");
  const paywall = OneTimePaywall.attach(contractAddress);

  // Check if the current signer is an admin
  const isAdmin = await paywall.hasRole(await paywall.DEFAULT_ADMIN_ROLE(), admin.address);
  if (!isAdmin) {
    console.log(" Error: Current signer is not an admin");
    return;
  }

  // Check if the new address is already an admin
  const isAlreadyAdmin = await paywall.hasRole(await paywall.DEFAULT_ADMIN_ROLE(), newAdmin.address);
  if (isAlreadyAdmin) {
    console.log(" Error: Address is already an admin");
    return;
  }

  console.log("Adding new admin...");
  const tx = await paywall.addAdmin(newAdmin.address);
  await tx.wait();

  // Verify the new admin was added
  const isNewAdmin = await paywall.hasRole(await paywall.DEFAULT_ADMIN_ROLE(), newAdmin.address);
  if (isNewAdmin) {
    console.log(" Successfully added new admin");
    console.log("Transaction hash:", tx.hash);
  } else {
    console.log(" Failed to add new admin");
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
}); 