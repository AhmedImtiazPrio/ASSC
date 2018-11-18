if [ ! -d "sleepEDFx" ]; then
  mkdir 'sleepEDFx'
fi
## cd 'sleepEDFx'
while IFS=, read -r col1
do
	echo "Downloading $col1"
	wget "https://physionet.org/pn4/sleep-edfx/$col1" -O "./sleepEDFx/$col1"
done < edfList.csv