# Zip files first
zip -q -r -X ./p1-vector-add.zip ./p1-vector-add

# Copy file to pinnalce
scp p1-vector-add.zip tgtracy@hpc-portal2.hpc.uark.edu:/home/tgtracy/project.zip

# Wait a sec because the host is bad
sleep 1

# From pinnacle, copy to hpc
ssh tgtracy@hpc-portal2.hpc.uark.edu "scp /home/tgtracy/project.zip tgtracy@login22:/home/tgtracy/project.zip"

# Copy to hpc
# scp -J tgtracy@pinnacle.uark.edu ./p1-vector-add tgtracy@login22:

# delete zip
rm -rf ./p1-vector-add.zip
