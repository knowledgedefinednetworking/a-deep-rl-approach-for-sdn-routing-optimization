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

#include "Application.h"


Define_Module(Application);


Application::Application() {

     interArrival = NULL;

}

Application::~Application() {
     cancelAndDelete(interArrival);
}


void Application::initialize()
{
    //id = extractId(this->getFullPath(), 12);
    id = par("id");
    genT = par("generation");
    lambdaFactor = par("lambda");
    dest = par("dest");
    MAXSIM = par("simulationDuration");
    numRx = par("numNodes");
    numPackets = 0;


    Statistic::instance()->setGeneration(genT);
    Statistic::instance()->setMaxSim(MAXSIM);
    Statistic::instance()->setLambda(lambdaFactor);


}


void Application::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {

        DataPacket *data = new DataPacket("dataPacket");

        int size;
        switch (genT) {
            case 0: // Poisson
                size = exponential(1000);
                if (size > 50000) size = 50000;
                break;
            case 1: // Deterministic
                size = 1000;
                break;
            case 2: // Uniform
                size = uniform(0,2000);
                break;
            case 3: // Binomial
                if (dblrand() < 0.5) size = 300;
                else size = 1700;
                break;
            default:
                break;
        }


        data->setBitLength(size);
        data->setTtl(numRx);

        data->setDstNode(dest);
        data->setSrcNode(id);
        data->setLastTS(simTime().dbl());

        send(data, "out");

        numPackets++;
        Statistic::instance()->setTraffic(simTime(), id, dest, size);
        Statistic::instance()->infoTS(simTime());


        if (simTime() < MAXSIM) {
            simtime_t etime= exponential(1.0/lambda);
            scheduleAt(simTime() + etime, msg);
        }
        else {
            EV << "END simulation" << endl;
        }
    }

    else {
        ControlPacket *data = check_and_cast<ControlPacket *>(msg);
        double flowRatio = data->getData();
        lambda = lambdaFactor*flowRatio;
        //lambda = lambdaMax/numRx;

        interArrival = new TimerNextPacket("timer");
        interArrival->setLambda(1.0/lambda);
        if (dest != id)
            scheduleAt(simTime() + 1.0/lambda, interArrival);
        ev << "Ratio: " << flowRatio << "   lambda: " << lambda << endl;

        delete data;


    }


}


/*
int Application::extractId(string name, int pos) {
    int idt;
    char c2 = name[pos+1]; char c1 = name[pos];
    if (c2 >= '0' and c2 <= '9')
        idt = (c1-'0')*10 + (c2 - '0');
    else
        idt = (c1-'0');
    return idt;
}*/
