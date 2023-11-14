# Zip files first
zip -q -r -X ./code.zip ./

# Copy file to pinnalce
scp code.zip tgtracy@hpc-portal2.hpc.uark.edu:/home/tgtracy/code.zip

# Wait a sec because the host is bad
# sleep 1

# From pinnacle, copy to hpc
ssh tgtracy@hpc-portal2.hpc.uark.edu "scp /home/tgtracy/code.zip tgtracy@login22:/home/tgtracy/code9.zip"

# Copy to hpc using a single command
# scp -J tgtracy@hpc-portal2.hpc.uark.edu code.zip tgtracy@login22:

# delete zip
rm -rf ./code.zip
