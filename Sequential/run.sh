if [ -f "metrics.csv" ]; then
    rm metrics.csv
    echo "HI"
fi

for i in {1..2}
do
    # ./a ../Dataset/karate.txt
    # ./a ../Dataset/dblp.txt
    # ./a ../Dataset/amazon.txt
    # ./a ../Dataset/roadNet-CA_OK.txt
    ./a ../Dataset/LiveJournal.txt
done