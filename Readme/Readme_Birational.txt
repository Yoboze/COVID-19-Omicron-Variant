#############################################################################################
#  How to run the code																		#
#############################################################################################

for job in 1 2 3 4 5; do python3 Birational.py > outputB_${job}.txt; done


#############################################################################################
#  How to generate the txt files															#
#############################################################################################


cat outputB_1.txt | grep -e 'daysB' | cut -f 2- -d ' ' > daysB.txt
cat outputB_1.txt | grep -e 'casesB' | cut -f 2- -d ' ' > casesB.txt
cat *.txt | grep -e 'timeB' | cut -f 2- -d ' ' > timeB.txt
cat *.txt | grep -e 'infectdB' | cut -f 2- -d ' ' > infectdB.txt
cat *.txt | grep -e 'alphaB' | cut -f 2- -d ' ' > alphaB.txt
cat *.txt | grep -e 'betaB:' | cut -f 2- -d ' ' > betaB.txt
cat *.txt | grep -e 'beta1B:' | cut -f 2- -d ' ' > beta1B.txt
cat *.txt | grep -e 'kappaB:' | cut -f 2- -d ' ' > kappaB.txt
cat *.txt | grep -e 'kappa1B:' | cut -f 2- -d ' ' > kappa1B.txt
cat *.txt | grep -e 'cB:' | cut -f 2- -d ' ' > cB.txt
cat *.txt | grep -e 'c1B:' | cut -f 2- -d ' ' > c1B.txt
cat *.txt | grep -e 'dB:' | cut -f 2- -d ' ' > dB.txt
cat *.txt | grep -e 'd1B:' | cut -f 2- -d ' ' > d1B.txt
cat *.txt | grep -e 'NfB:' | cut -f 2- -d ' ' > NfB.txt
cat *.txt | grep -e 'R_sqB:' | cut -f 2- -d ' ' > R_sqB.txt
cat *.txt | grep -e 'MAPE_B:' | cut -f 2- -d ' ' > MAPE_B.txt
cat *.txt | grep -e 'EV_B:' | cut -f 2- -d ' ' > EV_B.txt
cat *.txt | grep -e 'RMSE_B:' | cut -f 2- -d ' ' > RMSE_B.txt
cat *.txt | grep -e 'mse_train_lossB' | cut -f 2- -d ' ' > mse_train_lossB.txt
cat *.txt | grep -e 'rmse_train_lossB' | cut -f 2- -d ' ' > rmse_train_lossB.txt
cat *.txt | grep -e 'mse_validation_lossB' | cut -f 2- -d ' ' > mse_validation_lossB.txt
cat *.txt | grep -e 'rmse_validation_lossB' | cut -f 2- -d ' ' > rmse_validation_lossB.txt