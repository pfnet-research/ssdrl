cwd=$(pwd)
mkdir -p /tmp/ssdrl; cd /tmp
cp -r ${cwd}/* ssdrl/
rm ssdrl/make_zip_file_for_submission.sh
rm ~/reproducing_code.zip
zip -r ~/reproducing_code.zip ssdrl/* # Make zip file in home directory
rm -r ssdrl
cd $cwd