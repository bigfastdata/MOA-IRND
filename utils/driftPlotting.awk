##########################################################################
# 
# Process ARFF File and produce per-attribute graph of per-feature value drift
# Author: Brandon S. Parker <brandon.parker@utdallas.edu>
# Date: 2014-04-23
# Version: 3
# (c) 2013 Brandon S. Parker
#
# This process assumes you have GnuPlot installed and in the path. If not, the .gp file
# will not produce the PNG. You can run the .gp file once GnuPlot is obtained.
# The utility of this script is to view the behavior of the attributes in the data
# set over time, split out for each class/label. This shows the per-attribute
# linear separation and noise in the stream.
#
# Todo:
#  * Auto-detect for log scale plots if (max[attrib] > 4 * sqrt(abs((sum[attrib] / tally[attrib) ^ 2 - sumsum[attrib] / tally(attrib))))
#
#
##########################################################################

##########################################################################
# User defined functions
##########################################################################
function createHash(attributeName, className) {
	return attributeName"-"className;
}

function deducedPCA() {
	if (ARGV[fileArgNum] ~ /Netlogo/ ) {
		principleComponents["x"] = 3
		principleComponents["y"] = 4
		principleComponents["color"] = 6
		principleComponents["speed"] = 8
		autoDetectionSuccessfull = 2
		} 
	else if (ARGV[fileArgNum] ~ /SEA/ ) {
		principleComponents["a1"] = 1
		principleComponents["a2"] = 2
		principleComponents["a3"] = 3
		autoDetectionSuccessfull = 3
		}
	else if (ARGV[fileArgNum] ~ /Hyper/ ) {
		principleComponents["a1"]  = 1
		principleComponents["a2"]  = 2
		principleComponents["a3"]  = 3
		principleComponents["a4"]  = 4
		principleComponents["a5"]  = 5
		principleComponents["a6"]  = 6
		principleComponents["a7"]  = 7
		principleComponents["a8"]  = 8
		principleComponents["a9"]  = 9
		principleComponents["a10"] = 10
		autoDetectionSuccessfull = 4
		}
	else if (ARGV[fileArgNum] ~ /RBF/ ) {
		principleComponents["a1"]  = 1
		principleComponents["a2"]  = 2
		principleComponents["a3"]  = 3
		principleComponents["a4"]  = 4
		principleComponents["a5"]  = 5
		principleComponents["a6"]  = 6
		principleComponents["a7"]  = 7
		principleComponents["a8"]  = 8
		principleComponents["a9"]  = 9
		principleComponents["a10"] = 10
		autoDetectionSuccessfull = 5
		}
	else if (ARGV[fileArgNum] ~ /Agrawal/ ) {
			principleComponents["salary"] = 1
			principleComponents["commission"] = 2
			principleComponents["age"] = 3
			principleComponents["hval"] = 7
			principleComponents["hyears"] = 8
			principleComponents["loan"] = 9
			autoDetectionSuccessfull = 6
		}
	else if (ARGV[fileArgNum] ~ /PAMAP2/ ) {
			principleComponents["Timestamp"] = 1
			principleComponents["HeartRate"] = 2
			principleComponents["HandAccelX"] = 4
			principleComponents["HandAccelY"] = 5
			principleComponents["HandAccelZ"] = 6
			principleComponents["ChestTemp"] = 20
			principleComponents["ChestAccelX"] = 24
			principleComponents["ChestAccelY"] = 25
			principleComponents["ChestAccelZ"] = 26
			principleComponents["AnkleAccelX"] = 38 
			principleComponents["AnkleAccelY"] = 39
			principleComponents["AnkleAccelZ"] = 40
			autoDetectionSuccessfull = 7
		}
	else { # AutoDiscover
			print "WARNING: Attribute Auto-Discovery forced on..."
			autoDetectionSuccessfull = 0
		}
}

##########################################################################
# Initialization Block
##########################################################################
BEGIN {

# User-configurable settings:
windowSize = 5000
TMPDIR="./tmpdat"

# Do not change these settings:
useDiscoveredAttributes = 1
fileArgNum = 1
windowCur = 0
currentChunkNumber = 0
numFeature = 0
numClasses = 0
atData = false
FS=","
windowTally["strawman"][0] = 0;
windowSum["strawman"][0]   = 0;
windowLabels["Strawman"] = 0
autoDetectionSuccessfull = 0;
	tableOutput = "% Auto-generated data set table\n"
	tableOutput = tableOutput  "\\begin{table*}\n"
	tableOutput = tableOutput "\\centering\n"
	tableOutput = tableOutput  "\\caption[Data Set Characteristics]{Characteristics of the Data Sets used}\n"
	tableOutput = tableOutput  "\\label{tab:DataSetCharacteristics}\n"
	tableOutput = tableOutput  "\\begin{tabular}{|l||r|r|r|r|r|}\n"
	tableOutput = tableOutput  "\\hline\n"
	tableOutput = tableOutput  "DataSet      & \\# Classes & \\# Concurrent Classes & \\# Numeric Features & \\# Nominal Features & \\# Instances \\\\ \\hline\\hline\n"
	
GNUPlotFileName = "./DataFileAnalysisGraphs.gp"
delta=20
	dotsPerGraph=10
	print "# GNU Plot of feature value drift in data sets " 			 		>  GNUPlotFileName
	print "# Generated: " strftime() 											>> GNUPlotFileName
	print "reset"																>> GNUPlotFileName
	print "set tmargin 4"														>> GNUPlotFileName
	print "set term png large size 600,850	font 'arial,12'  "					>> GNUPlotFileName
	print "set datafile separator ','"											>> GNUPlotFileName
	
	print "set key  bmargin horizontal title 'Concept Labels:' enhanced box 0"  >> GNUPlotFileName
	print "set style data lp"													>> GNUPlotFileName
	print "set pointsize 1.5"													>> GNUPlotFileName
	print "set xlabel \"Data Instance Number\" font 'arial,14'"					>> GNUPlotFileName
	print "set ylabel \"Value\" font 'arial,14'"								>> GNUPlotFileName
	print "set format x \"%.1s %c\""											>> GNUPlotFileName
	print "set format y \"%.0f%%\""												>> GNUPlotFileName
	print "set xtics rotate by 45 offset -2,-1.6"								>> GNUPlotFileName
	
}

##########################################################################
# MAIN Block
##########################################################################
BEGINFILE {
	atData = false
	maxClassId = 0
	numDiscoveredAttributes  = 0
	TotalInstances = 0
	numClasses = 0
	numberOfActiveLabels = 0
	numNumericAttributes = 0	
	numNominalAttributes = 0
	currentChunkNumber = 0;
	FS=" "
}

/@RELATION|@relation/ {
	atData = false
	maxClassId = 0
	numDiscoveredAttributes  = 0
}

/@ATTRIBUTE|@attribute/ {
	atData = false
	attType = "unknown"
	numDiscoveredAttributes++
	if (($3 ~ "numeric") || ($3 ~ "Numeric") || ($3 ~ "NUMERIC")) {
		numNumericAttributes += 1
		attType = "Numeric"
	} else if ($3 !~ "[Cc]lass") {
		numNominalAttributes += 1
		attType = $3
		gsub("{","",attType)
		gsub("}","",attType)
		split(attType,attValList,",")
		for(val in attValList) {
			attributeReverseLookup[attValList[val]] = val
		}
		attType = "Nominal"
	}
	
	attributeNames[numDiscoveredAttributes] = $2
	attributeTypes[numDiscoveredAttributes] = attType
	if ($2 ~ "[Cc]lass") {
		numDiscoveredAttributes--
		classListString=$3
		gsub("{","",classListString)
		gsub("}","",classListString)
		split(classListString,classList,",")
		numClasses = 0
		for(k in classList) {
			LabelNames[k]=classList[k]
			LabelReverseLookup[classList[k]] = k
			numClasses++
		}
	}
}

/@DATA|@data/ {
	maxClassId = 0
	atData = true
	currentChunkNumber = 0;
	TotalInstances = 0
	FilenameBase = FILENAME
	gsub(/^[\.\\\/a-zA-Z0-9_]+\//,"",FilenameBase)
	gsub(/\.[a-zA-Z]{3,4}$/,"",FilenameBase)
	HMCSV = TMPDIR"/ClassHeatMap-"FilenameBase".csv"
	FVCSV = TMPDIR"/AttributeDrift-"FilenameBase"-"
	outline = "Chunk"
	for (j = 1; j <= numClasses; j++) {
		outline = outline", "LabelNames[j]
	}
	outline = outline
	print outline > HMCSV
	
	print "Starting to parse data section for file "FILENAME" as data set "FilenameBase" with "numClasses" labels and " numDiscoveredAttributes " features"
	FS=","
}

{ # Every line:
	if (FNR < 2) {
		atData = false
		currentChunkNumber = 0;
		windowCur = 0;
		delete windowSum
		delete windowTally
		delete windowLabels
		KnownFiles = FILENAME
	}
	classNum = LabelReverseLookup[$NF]
	if ((atData == true) && (classNum > 0) &&  (NF > 2) && ($1 !~ "@") ) {
		
		
		TotalInstances += 1
		#gsub("[a-zA-Z_]","",classNum)
		for(i = 1; i < NF; i++) {
			val = $i
			if (attributeTypes[i] == "Nominal") {
				val = sprintf("%.1f",attributeReverseLookup[$i])
			}
			windowSum[i][classNum] += val
			windowTally[i][classNum] += 1
			windowLabels[classNum] += 1
		}
		knownClasses[classNum] += 1
		windowCur++
		if (windowCur >= windowSize) {
			outline = currentChunkNumber
			for (j = 1; j <= numClasses; j++) {
				outline = outline ", "sprintf("%i",windowLabels[j])
			}
			outline = outline
			print outline >> HMCSV
			numLabelsInThisWindow = 0
			for(t in windowLabels) { 
				if (windowLabels[t] > 0) {
					numLabelsInThisWindow++
				}
			}
			if (numberOfActiveLabels < numLabelsInThisWindow) {
				numberOfActiveLabels = numLabelsInThisWindow
			}
			if (currentChunkNumber  < 1) {
				 for(a = 1; a < numDiscoveredAttributes; a++) {
					outline = "Chunk"
					for (c = 1; c <= numClasses; c++) {
						outline = outline", "LabelNames[c]
					}
					print outline > FVCSV a ".csv"
				 }
			}
			for(a = 1; a <= numDiscoveredAttributes; a++) {
				outline = currentChunkNumber
				for (c = 1; c < numClasses; c++) {
					if (windowTally[a][c] > 0) {
						outline = outline", "sprintf("%.8f",windowSum[a][c] / windowTally[a][c])
					} else {
						outline = outline", "sprintf("%.8f",0.0)
					}
				}
				outline = outline
				print outline >> FVCSV a ".csv"
			 }
			
			currentChunkNumber++;
			windowCur = 0;
			delete windowSum
			delete windowTally
			delete windowLabels
		} # EndIf Window.End()
	}
}


##########################################################################
# EOF Processing Block (Gawk specific)
##########################################################################
ENDFILE {
	print "Done parsing file "FilenameBase" after "currentChunkNumber" chunks and "TotalInstances " instances"
	actualSeenClasses = 0
	for(tmp in knownClasses) {
		actualSeenClasses++
	}
	# Tables:
	tableOutput = tableOutput FilenameBase "& " actualSeenClasses " & " numberOfActiveLabels " & " numNumericAttributes " & " numNominalAttributes " & " TotalInstances" \\\\ \\hline\n"
	
	# Heatmap GnuPlot:
	print "set term png large size 800,400	font 'arial,12' "	>> GNUPlotFileName
	print "unset key"															>> GNUPlotFileName
	print "unset logscale"														>> GNUPlotFileName
	print "set pointsize 1"														>> GNUPlotFileName
	print "set xlabel \"Data Instance Number\" font 'arial,14'"					>> GNUPlotFileName
	print "set ylabel \"Value\" font 'arial,14'"								>> GNUPlotFileName
	print "set format x \"%.1s %c\""											>> GNUPlotFileName
	print "set format y \"%s\"" 												>> GNUPlotFileName
	print "set xtics rotate by 0 offset 0,0"									>> GNUPlotFileName
	print "set xrange [ 0.0 : "(currentChunkNumber - 1)" ]"						>> GNUPlotFileName
	print "set yrange [ 1.0 : "numClasses" ]"									>> GNUPlotFileName
	print "unset xrange "											>> GNUPlotFileName
	print "unset yrange"											>> GNUPlotFileName
	print "set autoscale "											>> GNUPlotFileName
	print "set style fill transparent solid 0.2 noborder"						>> GNUPlotFileName
	print "set output \"./graphs/LabelOccuranceHeatmap-"FilenameBase".png\""				>> GNUPlotFileName
	
	print "plot \\"	>> GNUPlotFileName
	i = 0
	for (i = 2; i < numClasses + 2; i++) {
		comma = " "
		if (i > 2) { comma = ","}
		#print comma"\""HMCSV"\" using 1:"i":2:($"i") title \""attributeReverseLookup[i]"\" w circle lc var \\" 	>> GNUPlotFileName;
		print comma"\""HMCSV"\" using 1:"i":2:($"i") w circle lc var \\" 	>> GNUPlotFileName;
	}
	print "" >> GNUPlotFileName
	
#	print "plot  '"HMCSV"' u 1:2($2): w circle lc var" >> GNUPlotFileName
	
	
	
	print "unset output"													>> GNUPlotFileName
	
	
	
	# Attribute Drift Plots:
	for(a = 1; a <= numDiscoveredAttributes; a++) {
		GPDataFile = FVCSV a ".csv"
		
	}
	
	
	
	
	TotalInstances = 0
	numClasses = 0
	numberOfActiveLabels = 0
	numNumericAttributes = 0	
	numNominalAttributes = 0
	delete knownClasses
	delete attributeNames
	delete attributeTypes
	delete LabelNames
}


#####
#
#  Final End of all processing
#
#####
END {
	print "exit gnuplot"														>> GNUPlotFileName
	print "Compiling graph files..."
	#system("gnuplot < " GNUPlotFileName);
	
	tableOutput = tableOutput "\\end{tabular}\n\\end{table*}"
	print tableOutput
	print "%Done."
}
