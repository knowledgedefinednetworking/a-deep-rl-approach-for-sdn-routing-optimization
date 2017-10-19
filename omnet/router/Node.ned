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

package networksimulator;

module Node
{
    parameters:
        int numTx;
        int numNodes;
        int id;

        @display("i=misc/node_vs,gold;bgb=224,246");




    gates:
        inout port[];

    submodules:
        tcontroller: TrafficController {
            id = id;
            nodeRatio = uniform(0.1,1);
            numNodes = numTx;
        };
        app[numNodes]: Application {
            id = id;
            dest = index;
            numNodes = numTx;
        };
        routing: Balancer {
            numTx = numTx;
            numNodes = numTx;
            id = id;
        };
        queue[sizeof(port)]: NodeQueue;

    connections:
        for i=0..numNodes-1 {
            routing.localIn++ <-- app[i].out;
            tcontroller.out++ --> app[i].tcontrol;
        }
        for i=0..sizeof(port)-1 {
            routing.out++ --> queue[i].in;
            routing.in++ <-- queue[i].out;
            queue[i].line <--> port[i];
        }
}
