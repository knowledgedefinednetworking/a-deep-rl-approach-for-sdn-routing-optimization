//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

#include "TrafficController.h"


Define_Module(TrafficController);

void TrafficController::initialize()
{
    id = par("id");
    nodeRatio = par("nodeRatio");
    numNodes = par("numNodes");
    folderName = par("folderName").stdstringValue();

    int MODE = 3;

    if (MODE == 1) {
        // UNIFORM TRAFFIC PER FLOW
        for (int i = 0; i < numNodes; i++) {
            double aux;
            if (i == id) aux = 0;
            else aux = uniform(0.1,1);

            ControlPacket *data = new ControlPacket("trafficInfo");
            data->setData(aux/numNodes);
            send(data, "out", i);
        }
    }
    else if (MODE == 2) {
        // UNIFOR TRAFFIC PER NODE
        double flowRatio[numNodes];
        double sumVal = 0;
        for (int i = 0; i < numNodes; i++) {
            double aux;

            if (i == id) aux = 0;
            else aux = uniform(0.1,1);

            flowRatio[i] = aux;
            sumVal += aux;
        }

        //ev << "NODE Ratio: " << nodeRatio << endl;
        for (int i = 0; i < numNodes; i++) {
            ControlPacket *data = new ControlPacket("trafficInfo");
            data->setData(nodeRatio*flowRatio[i]/sumVal);
            send(data, "out", i);
        }
    }
    else {
        // READED FROM FILE
        getTrafficInfo(id, flowRatio);
        for (int i = 0; i < numNodes; i++) {
            ControlPacket *data = new ControlPacket("trafficInfo");
            data->setData(flowRatio[i]);
            send(data, "out", i);
        }
    }

}

void TrafficController::handleMessage(cMessage *msg)
{
    // TODO - Generated method body
}

void TrafficController::getTrafficInfo(int id, double rData[]) {

     string line;
     ifstream myfile (folderName + "/Traffic.txt");
     double val;

     if (myfile.is_open()) {
         int i = 0;
         while (id != i) {
             for(int k = 0; k < numNodes; k++) {
                 string aux;
                 getline(myfile, aux, ',');
             }
             //myfile >> val;
             i++;
         }

         for(int k = 0; k < numNodes; k++) {
             string aux;
             getline(myfile, aux, ',');
             val = stod(aux);
             rData[k] = val;
         }

         myfile.close();
     }
}
