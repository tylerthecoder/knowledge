# Zip files first
zip -q -r -X ./p2.zip ./

# Copy file to pinnalce
scp p2.zip tgtracy@hpc-portal2.hpc.uark.edu:/home/tgtracy/project.zip

# Wait a sec because the host is bad
sleep 1

# From pinnacle, copy to hpc
ssh tgtracy@hpc-portal2.hpc.uark.edu "scp /home/tgtracy/project.zip tgtracy@login22:/home/tgtracy/project.zip"

# delete zip
rm -rf ./p2.zip
