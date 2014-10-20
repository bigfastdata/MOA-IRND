@echo off
REM Base Config: EvaluateNonStationaryDynamicStream -L novelClass.M3Ballot -s generators.InducedRandomNonStationaryDataGenerator -e ClassificationWithNovelClassPerformanceEvaluator

set MEMORY=10G
set OPTIMIZATIONS=-d64 -XX:+AggressiveOpts -server -XX:+UseCompressedOops -XX:+UseBiasedLocking
set DEBUGOPTIONS=""
REM set DEBUGOPTIONS=-XX:-PrintClassHistogram -XX:-TraceClassLoading
set IRNDJAR=.\bin\irnd.jar
set SLBXJAR=.\bin\Sluicebox.jar
REM set OTHERJARS=.\extensions\iovfdt.jar;.\extensions\EnsembleClassifiers.jar
REM set OTHERJARS=.\extensions\iovfdt.jar
set OTHERJARS=.\bin\EnsembleClassifiers.jar
set MOAJAR=.\bin\moa-2013.12.jar
set WEKAJAR=.\bin\weka-dev-3.7.10.jar
set MOASZJAR=.\bin\sizeofag-1.0.0.jar
set MOACMD=java %OPTIMIZATIONS% -Xmx%MEMORY% -cp %IRNDJAR%;%OTHERJARS%;%WEKAJAR%;%MOAJAR% -javaagent:%MOASZJAR% moa.DoTask

set EVALCMD=EvaluateNonStationaryDynamicStream
set FREQ=5000
set CHUNKSIZE=1
set DATADIR=data
set LATENCY=500
set WARMUP=5000
set DEADLINE=100
set OUTDIR=results
set DATAFILES=FC KDD99 PAMAP2 IRND5M SEA Hyperplane RBF Agrawal Waveform STAGGER RandTree
set LEARNERS=meta.M3 meta.LeveragingBag macros.TACNB meta.WeightedMajorityAlgorithm meta.TemporallyAugmentedClassifier  bayes.NaiveBayes functions.Perceptron functions.MajorityClass functions.NoChange trees.DecisionStump functions.RandomGuess
REM These two take a LONG time to run...
REM set LEARNERS=meta.DynamicWeightedMajority
set PERCENTAGES=1.0 0.1 0.01 0.001 0.0001 0.00001
set MAXGENSIZE=50000000

mkdir %DATADIR%
mkdir %OUTDIR%
REM - takes care of most cases
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO FOR %%L IN (%LEARNERS%) DO %MOACMD% "%EVALCMD% -L %%L -s (ArffFileStream -f %DATADIR%\%%D.arff ) -T %DEADLINE% -w %WARMUP% -l %LATENCY% -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_%%L_%%P.csv -O  %OUTDIR%\%%D_%%L_%%P.out

REM Leverage Bagging Variations
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.LeveragingBag -m LeveragingBagME) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_LBME_%%P.csv -O  %OUTDIR%\%%D_LBME_%%P.out
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.LeveragingBag -m LeveragingBagWT) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_LBWT_%%P.csv -O  %OUTDIR%\%%D_LBWT_%%P.out
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.LeveragingBag -m LeveragingBagHalf) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_LBHalf_%%P.csv -O  %OUTDIR%\%%D_LBHalf_%%P.out
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.LeveragingBag -m LeveragingSubag) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_LBSub_%%P.csv -O  %OUTDIR%\%%D_LBSub_%%P.out
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.LeveragingBag -l meta.M3) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_LBM3_%%P.csv -O  %OUTDIR%\%%D_LBM3_%%P.out

REM LearneNSE++ Variations
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.LearnNSE -s ERROR) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_LNSEe_%%P.csv -O  %OUTDIR%\%%D_LNSEe_%%P.out
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.LearnNSE -s AGE) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_LNSEa_%%P.csv -O  %OUTDIR%\%%D_LNSEa_%%P.out

REM AWE Takes a lot of ram and a few hours to run
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.AccuracyWeightedEnsemble) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_AWE_%%P.csv -O  %OUTDIR%\%%D_AWE_%%P.out


REM DWM takes a LONG Time to run, so do it last
FOR %%P IN (%PERCENTAGES%) DO FOR %%D IN (%DATAFILES%) DO %MOACMD% "%EVALCMD% -L (meta.DynamicWeightedMajority -e 100) -w %WARMUP% -l %LATENCY% -s (ArffFileStream -f %DATADIR%\%%D.arff ) -i %MAXGENSIZE% -p %%P -f %FREQ% " -c %CHUNKSIZE% -d %OUTDIR%\%%D_DWM100_%%P.csv -O  %OUTDIR%\%%D_DWM100_%%P.out


echo "DONE."