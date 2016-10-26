
#include <fstream>
#include <iostream>
#include <sstream>


#include "pugixml.hpp"
using namespace pugi;


using namespace std;

template <typename T>
string numToStrHM(T number)
{
    ostringstream s;
    s << number;
    return s.str();
}

double startThresholdHM , endThresholdHM , intervalThresholdHM ,negGenMUHM;
double startNegWeight ,endNegWeight ,negWeightInterval;
double startNegMu, endNegMu, NegMuInterval;
double startDelta, endDelta, deltaInterval;

double smoothJMInterval1, smoothJMInterval2;

int RSMethodHM; // 0--> LM , 1--> RecSys
int negGenModeHM;//0 --> coll , 1--> nonRel

int feedbackMode;//0 --> no fb, 1-->ours , 2-->normal , 3-->mixture

int updatingThresholdMode;//0 -> no updating, 1->linear , 2->diffAlpha

int WHO;// 0--> server , 1-->Mozhdeh, 2-->AP, other-->Hossein
string outputFileNameHM;
string resultFileNameHM;
void readParamFile(string paramFileName);
void readParams(string paramFileName)
{
    readParamFile(paramFileName);

    if(RSMethodHM==1)
    {
        if(negGenModeHM == 0)
        {

            outputFileNameHM = "out/NegColl_";
            resultFileNameHM ="res/NegColl_";
        }else if(negGenModeHM == 1)
        {
            outputFileNameHM += "out/NegNonRel_";
            resultFileNameHM += "res/NegNonRel_";
        }
        else if(negGenModeHM == 2)
        {
            outputFileNameHM += "out/UniNegNonRel_";
            resultFileNameHM += "res/UniNegNonRel_";
        }
        else if(negGenModeHM == 3)
        {
            outputFileNameHM += "out/SmoothUniNegNonRel_";
            resultFileNameHM += "res/SmoothUniNegNonRel_";
        }
        else if(negGenModeHM == 4)
        {
            outputFileNameHM += "out/RelNegCol_";
            resultFileNameHM += "res/RelNegCol_";
        }
        //outputFileNameHM+=numToStrHM(negGenMUHM)+"_";
        //resultFileNameHM+=numToStrHM(negGenMUHM)+"_";
    }else if (RSMethodHM == 0)
    {
        outputFileNameHM += "out/LM_";
        resultFileNameHM += "res/LM_";
    }
    else if(RSMethodHM == 2){
        outputFileNameHM += "out/NegKLQTE_";
        resultFileNameHM += "res/NegKLQTE_";

    }
    else if(RSMethodHM == 3){
        outputFileNameHM += "out/NegKL_";
        resultFileNameHM += "res/NegKL_";

    }
    else if(RSMethodHM == 4){
        outputFileNameHM += "out/Fang_";
        resultFileNameHM += "res/Fang_";

    }
    if(feedbackMode == 0)//no fb
    {
        outputFileNameHM+="Nofb_";
        resultFileNameHM += "Nofb_";
    }else if(feedbackMode == 1)//ours
    {
        outputFileNameHM+="NegFB_negWeight:_"+numToStrHM(startNegWeight)+":"+numToStrHM(endNegWeight)+"("+numToStrHM(negWeightInterval)+")_";
        resultFileNameHM += "NegFB_";
    }else if(feedbackMode == 2)//normal feedback
    {
        outputFileNameHM+="NormalFB_";
        resultFileNameHM += "NormalFB_";
    }
    else if(feedbackMode == 3)//mixture feedback
    {
        outputFileNameHM+="MixtureFB_";
        resultFileNameHM += "MixtureFB_";
    }
    else if(feedbackMode == 4)//mixture feedback
    {
        outputFileNameHM+="NegMixtureFB_";
        resultFileNameHM += "NegMixtureFB_";
    }

    if(updatingThresholdMode == 0)//no updating
    {
        outputFileNameHM +="NoUpdatingThr_";
        resultFileNameHM += "NoUpdatingThr_";

    }else if(updatingThresholdMode == 1)//linear
    {
        outputFileNameHM+="LinearUpdatingThr_";
        resultFileNameHM += "LinearUpdatingThr_";
    }else if(updatingThresholdMode == 2)
    {
        outputFileNameHM+="DiffAlphaUpdatingThr_";
        resultFileNameHM += "DiffAlphaUpdatingThr_";
    }

    outputFileNameHM += "profDocThr:_"+numToStrHM(startThresholdHM)+":"+numToStrHM(endThresholdHM)+"("+numToStrHM(intervalThresholdHM)+")";
    outputFileNameHM += "NegMu:_"+numToStrHM(startNegMu)+":"+numToStrHM(endNegMu)+"("+numToStrHM(NegMuInterval)+")";
    outputFileNameHM += "Delta:_"+numToStrHM(startDelta)+":"+numToStrHM(endDelta)+"("+numToStrHM(deltaInterval)+")";
    outputFileNameHM += "Lambda1:_"+numToStrHM(smoothJMInterval1);

}


void readParamFile(string paramfileName)
{
    xml_document doc;
    xml_parse_result result = doc.load_file(paramfileName.c_str());// Your own path to original format of queries
    if(result.status != 0)
        cerr<<"\n\nCANT READ PARAM FILE!!!\n\n";
    xml_node topics = doc.child("parameters");

    for (xml_node_iterator topic = topics.begin(); topic != topics.end(); topic++)
    {
        istringstream iss ;
        iss.str(topic->child("Who").first_child().value());
        iss>>WHO;

        iss.clear();
        iss.str ( topic->child("RetMethod").first_child().value());
        iss >> RSMethodHM ;

        iss.clear();
        iss.str( topic->child("NegativeMode").first_child().value());
        iss>>negGenModeHM;
        
        iss.clear();
        iss.str( topic->child("SmoothJMInterval1").first_child().value());
        iss>>smoothJMInterval1;

        iss.clear();
        iss.str( topic->child("SmoothJMInterval2").first_child().value());
        iss>>smoothJMInterval2;

      //  iss.clear();
       // iss.str(topic->child("NegMu").first_child().value());
        //iss>>negGenMUHM;
        iss.clear();
        iss.str( topic->child("StartNegMu").first_child().value());
        iss>>startNegMu;

        iss.clear();
        iss.str( topic->child("EndNegMu").first_child().value());
        iss>>endNegMu;

        iss.clear();
        iss.str( topic->child("NegMuInterval").first_child().value());
        iss>>NegMuInterval;

        iss.clear();
        iss.str( topic->child("StartDelta").first_child().value());
        iss>>startDelta;

        iss.clear();
        iss.str( topic->child("EndDelta").first_child().value());
        iss>>endDelta;

        iss.clear();
        iss.str( topic->child("DeltaInterval").first_child().value());
        iss>>deltaInterval;

        iss.clear();
        iss.str( topic->child("StartThr").first_child().value());
        iss>>startThresholdHM;

        iss.clear();
        iss.str( topic->child("EndThr").first_child().value());
        iss>>endThresholdHM;

        iss.clear();
        iss.str( topic->child("ProfDocSimInterval").first_child().value());
        iss>>intervalThresholdHM;

        iss.clear();
        iss.str( topic->child("FBMode").first_child().value());
        iss>>feedbackMode;

        iss.clear();
        iss.str( topic->child("StartNegWeight").first_child().value());
        iss>>startNegWeight;

        iss.clear();
        iss.str( topic->child("EndNegWeight").first_child().value());
        iss>>endNegWeight;

        iss.clear();
        iss.str( topic->child("NegWeightInterval").first_child().value());
        iss>>negWeightInterval;

        iss.clear();
        iss.str( topic->child("UpdatingThrMode").first_child().value());
        iss>>updatingThresholdMode;

//        cout<<WHO<<RSMethodHM<<negGenModeHM<<negGenMUHM<<startThresholdHM<<endThresholdHM<<intervalThresholdHM<<feedbackMode<<
//            startNegWeight<<endNegWeight<<  negWeightInterval<<updatingThresholdMode;

    }
}
