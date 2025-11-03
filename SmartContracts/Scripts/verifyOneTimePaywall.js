const hre = require("hardhat");

async function main() {
  const contractAddress = "0x44A4e524d55aB9D7c313d3c0cCDaB64c7C7D4AA9"; 

  console.log("Verifying contract at:", contractAddress);
  
  try {
    await hre.run("verify:verify", {
      address: contractAddress,
      constructorArguments: [], 
    });
    console.log("Contract verified successfully");
  } catch (error) {
    if (error.message.includes("Already Verified")) {
      console.log("Contract is already verified!");
    } else {
      console.error("Verification failed:", error);
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 