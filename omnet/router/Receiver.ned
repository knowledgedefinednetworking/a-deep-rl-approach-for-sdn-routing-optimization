
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

module Receiver
{
    parameters:
        int id;
        int numTx;
        int numNodes;
        @display("i=misc/node_vs,gold;bgb=224,246");




    gates:
        inout port[];

    submodules:
        routing: Routing {
            numTx = numTx;
            numNodes = numNodes;
            id = id;
        };
        queue[sizeof(port)]: NodeQueue;

    connections:

        for i=0..sizeof(port)-1 {
            routing.out++ --> queue[i].in;
            routing.in++ <-- queue[i].out;
            queue[i].line <--> port[i];
        }
}
