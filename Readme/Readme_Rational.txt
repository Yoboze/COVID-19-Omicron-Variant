#############################################################################################
#  How to run the code																		#
#############################################################################################

for job in 1 2 3 4 5; do python3 Rational.py > outputR_${job}.txt; done

#############################################################################################
#  How to generate the txt files															#
#############################################################################################


cat outputR_1.txt | grep -e 'daysR' | cut -f 2- -d ' ' > daysR.txt
cat outputR_1.txt | grep -e 'casesR' | cut -f 2- -d ' ' > casesR.txt
cat *.txt | grep -e 'timeR' | cut -f 2- -d ' ' > timeR.txt
cat *.txt | grep -e 'infectdR' | cut -f 2- -d ' ' > infectdR.txt
cat *.txt | grep -e 'alphaR' | cut -f 2- -d ' ' > alphaR.txt
cat *.txt | grep -e 'betaR:' | cut -f 2- -d ' ' > betaR.txt
cat *.txt | grep -e 'kappaR:' | cut -f 2- -d ' ' > kappaR.txt
cat *.txt | grep -e 'NfR:' | cut -f 2- -d ' ' > NfR.txt
cat *.txt | grep -e 'dR:' | cut -f 2- -d ' ' > dR.txt
cat *.txt | grep -e 'R_sqR:' | cut -f 2- -d ' ' > R_sqR.txt
cat *.txt | grep -e 'MAPE_R:' | cut -f 2- -d ' ' > MAPE_R.txt
cat *.txt | grep -e 'EV_R:' | cut -f 2- -d ' ' > EV_R.txt
cat *.txt | grep -e 'RMSE_R:' | cut -f 2- -d ' ' > RMSE_R.txt
cat *.txt | grep -e 'mse_train_lossR' | cut -f 2- -d ' ' > mse_train_lossR.txt
cat *.txt | grep -e 'rmse_train_lossR' | cut -f 2- -d ' ' > rmse_train_lossR.txt
cat *.txt | grep -e 'mse_validation_lossR' | cut -f 2- -d ' ' > mse_validation_lossR.txt
cat *.txt | grep -e 'rmse_validation_lossR' | cut -f 2- -d ' ' > rmse_validation_lossR.txt