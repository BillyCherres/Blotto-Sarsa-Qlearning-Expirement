from blottoGeneral import universalBlotto
# 0 = Random Agent
# 1 = SARSA Agent
# 2 = QL Agent


#sarsaVSql = universalBlotto(player1=1, player2=2,simulations=1)
#sarsaVSql.playSim()
#sarsaVSql.plotMultipleGraphs()
#sarsaVSql.plotAverageGraph()

#sarsaVSql1 = universalBlotto(player1=1, player2=2,simulations=1)
#sarsaVSql1.playSim()
#sarsaVSql1.plotMultipleGraphs()

#QvR = universalBlotto(player1=2, player2=0)
#QvR.playSim()
#QvR.plotMultipleGraphs()
#QvR.postTrainingAnalysis()

RvQ = universalBlotto(player1=2, player2=2)
RvQ.playSim()
RvQ.plotMultipleGraphs()

blottoGames = []

blottoGames.append(universalBlotto(player1=0, player2= 0)) # RvR
blottoGames.append(universalBlotto(player1=2, player2= 0)) # QvR
blottoGames.append(universalBlotto(player1=1, player2= 0)) # SvR
blottoGames.append(universalBlotto(player1=2, player2= 2)) # QvQ
blottoGames.append(universalBlotto(player1=2, player2= 1)) # QvS
blottoGames.append(universalBlotto(player1=1, player2= 1)) # SvS

for i in range(len(blottoGames)):
    blottoGames[i].playSim()
    blottoGames[i].plotMultipleGraphs()