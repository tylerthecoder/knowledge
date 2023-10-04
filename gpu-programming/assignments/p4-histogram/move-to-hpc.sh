# Zip files first
zip -q -r -X ./code4.zip ./

# Copy file to pinnalce
scp code4.zip tgtracy@hpc-portal2.hpc.uark.edu:/home/tgtracy/code4.zip

# Wait a sec because the host is bad
sleep 1

# From pinnacle, copy to hpc
ssh tgtracy@hpc-portal2.hpc.uark.edu "scp /home/tgtracy/code4.zip tgtracy@login22:/home/tgtracy/code4.zip"

# Copy to hpc using a single command
# scp -J tgtracy@pinnacle.uark.edu ./p1-vector-add tgtracy@login22:

# delete zip
rm -rf ./code.zip
