#############################################################################################
#  How to run the code																		#
#############################################################################################

for ((i=0; i<5; i++)); do python3 Time_series.py > outputA_${i}.txt; done

OR

for job in 1 2 3 4 5; do python3 Time_series.py > outputA_${job}.txt; done

#############################################################################################
#  How to generate the txt files															#
#############################################################################################


cat outputA_1.txt | grep -e 'daysA' | cut -f 2- -d ' ' > daysA.txt
cat outputA_1.txt | grep -e 'casesA' | cut -f 2- -d ' ' > casesA.txt
cat *.txt | grep -e 'timeA' | cut -f 2- -d ' ' > timeA.txt
cat *.txt | grep -e 'infectdA' | cut -f 2- -d ' ' > infectdA.txt
cat *.txt | grep -e 'alphaA' | cut -f 2- -d ' ' > alphaA.txt
cat *.txt | grep -e 'Nf_A:' | cut -f 2- -d ' ' > Nf_A.txt
cat *.txt | grep -e 'R_sqA:' | cut -f 2- -d ' ' > R_sqA.txt
cat *.txt | grep -e 'MAPE_A:' | cut -f 2- -d ' ' > MAPE_A.txt
cat *.txt | grep -e 'EV_A:' | cut -f 2- -d ' ' > EV_A.txt
cat *.txt | grep -e 'RMSE_A:' | cut -f 2- -d ' ' > RMSE_A.txt
cat *.txt | grep -e 'mse_train_lossA' | cut -f 2- -d ' ' > mse_train_lossA.txt
cat *.txt | grep -e 'rmse_train_lossA' | cut -f 2- -d ' ' > rmse_train_lossA.txt
cat *.txt | grep -e 'mse_validation_lossA' | cut -f 2- -d ' ' > mse_validation_lossA.txt
cat *.txt | grep -e 'rmse_validation_lossA' | cut -f 2- -d ' ' > rmse_validation_lossA.txt