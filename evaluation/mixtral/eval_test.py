import sys
sys.path.insert(1, '../ollama/ollama-python/')
import ollama

from ollama import Client
client = Client(host="127.0.0.1:11434")
model = "mixtral"
system_prompt = """
Imagine you are a robot assistance AI. You help to manage a mobile ground rescue robot. Use the given task description to create an optimal set of actions which fulfills the given task.

Create the optimal set of actions needed using the scenario description and the following action pattern:

[{'type': 'type', 'pose': {'x': 'x-position', 'y':'y-position'}, 'costs': 0.0, 'execution_mode': 'execution_mode', 'entity': {'entity_type': 'entity_type', 'entity_super_type': 'entity_super_type', 'id': 'id', 'parent_id': 'parent_id'}}]

Type is the type of the action and can be goto, measure_hazard, inspect, search, communicate, or traverse.
Pose is the pose where the robot is after executing the action. Execution_mode is either teleop, autonomous or assisted and describes wether the action is executed teleoperated, assisted or autonomous by the robot. Costs are the costs needed by the robot to execute the action in the specified mode. Entity is the entity on which the robot executes the action. The entity_type can either be a room or object of the environment or a generic point of interest (poi). The entity_super_type is either room, object or poi. The id is the id of the entity in the environment. The parent_id is the id of the room in which the robot is after executing the action.

Here are some examples for a scenario in with two rooms. Room 1 has its center at position 5, 5 and Room 2 has its center at 10, 5. A door with id 1 connects the rooms at 7,5. In room 1 at 2,2 is a victim with id 2. In room 2 at 10, 4 is a box with id 3 and also a cupboard with id 4 at 10, 8. 
Rooms, doors, victims, boxes and cupboards can be reached with the goto action. Room traversals with doors have to be considered. The costs are the euclidean distance needed to reach the entity regardless of the execution mode. 
Doors can be traversed to go from one room to the other. For this the robot must be at the doors position. The cost to do this is 40 for autonomous, 30 for assisted and 20 for teleop.
The robot can communicate with victims if he is at his position. The cost for this is 60 when teleoperated. This can only be done teleoperated.
Boxes and cupboards can be inspected for a cost of this is 40 teleop, 35 autonomous and 30 assisted. Additionally boxes and cupboards can be searched for a cost of 40 teleop, 35 autonomous and 30 assisted.
The robot starts at 0,0. If a task is not possible, clearly state that. This goes for the case that an entity is not reachable. If not enough information was provided also state that.

Example: 'go to the victim 2'
Result: [{'type': 'goto', 'pose': {'x': 2, 'y': 2}, 'costs': 2.8284, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'victim', 'entity_super_type': 'object', 'id': 2, 'parent_id': 1}}]

Example: 'go to the victim in room 1'
Result: [{'type': 'goto', 'pose': {'x': 2, 'y': 2}, 'costs': 2.8284, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'victim', 'entity_super_type': 'object', 'id': 2, 'parent_id': 1}}]

Example: 'communicate with victim 2'
Result: [{'type': 'goto', 'pose': {'x': 2, 'y': 2}, 'costs': 2.8284, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'victim', 'entity_super_type': 'object', 'id': 2, 'parent_id': 1}},
{'type': 'communicate', 'pose': {'x': 2, 'y': 2}, 'costs': 60, 'execution_mode': 'teleop', 'entity': {'entity_type': 'victim', 'entity_super_type': 'object', 'id': 2, 'parent_id': 1}}]

Example: 'traverse door 1'
Result: [{'type': 'goto', 'pose': {'x': 7, 'y': 5}, 'costs': 8.6023, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 1}},
{'type': 'traverse', 'pose': {'x': 7, 'y': 5}, 'costs': 20, 'execution_mode': 'teleop', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 2}}]

Example: 'go to room 2'
Result: [{'type': 'goto', 'pose': {'x': 7, 'y': 5}, 'costs': 8.6023, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 1}},
{'type': 'traverse', 'pose': {'x': 7, 'y': 5}, 'costs': 20, 'execution_mode': 'teleop', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 2}},
{'type': 'goto', 'pose': {'x': 10, 'y': 5}, 'costs': 3, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'room', 'entity_super_type': 'room', 'id': 2, 'parent_id': 2}}]

Example: 'inspect box 3'
Result: [{'type': 'goto', 'pose': {'x': 7, 'y': 5}, 'costs': 8.6023, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 1}},
{'type': 'traverse', 'pose': {'x': 7, 'y': 5}, 'costs': 20, 'execution_mode': 'teleop', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 2}},
{'type': 'goto', 'pose': {'x': 2, 'y': 2}, 'costs': 3.162, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'box', 'entity_super_type': 'object', 'id': 3, 'parent_id': 1}},
{'type': 'communicate', 'pose': {'x': 10, 'y': 4}, 'costs': 30, 'execution_mode': 'assisted', 'entity': {'entity_type': 'box', 'entity_super_type': 'object', 'id': 3, 'parent_id': 1}}]

Example: 'traverse door 1 and go to box 3'
Result: [{'type': 'goto', 'pose': {'x': 7, 'y': 5}, 'costs': 8.6023, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 1}},
{'type': 'traverse', 'pose': {'x': 7, 'y': 5}, 'costs': 20, 'execution_mode': 'teleop', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 2}},
{'type': 'goto', 'pose': {'x': 2, 'y': 2}, 'costs': 3.162, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'box', 'entity_super_type': 'object', 'id': 3, 'parent_id': 1}}]

Example: 'go to room 2. Don't open doors teleop'
Result: [{'type': 'goto', 'pose': {'x': 7, 'y': 5}, 'costs': 8.6023, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 1}},
{'type': 'traverse', 'pose': {'x': 7, 'y': 5}, 'costs': 30, 'execution_mode': 'assisted', 'entity': {'entity_type': 'door', 'entity_super_type': 'object', 'id': 1, 'parent_id': 2}},
{'type': 'goto', 'pose': {'x': 10, 'y': 5}, 'costs': 3, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'room', 'entity_super_type': 'room', 'id': 2, 'parent_id': 2}}]

Example: 'go to the fire_extinguisher'
Result: ERROR, There is not fire_extinguisher

Example: 'go to box 4'
Result: ERROR, There is no box 4, only box 3

Example 'communicate with box 3'
Result: ERROR, I can't communicate with a box, only goto, inspect or search are possible

Example: 'go to the box, but don't traverse door 1'
Result: ERROR, There is no valid path to reach box 3

Example: 'go to the victim, but don't inspect boxes'
Result: [{'type': 'goto', 'pose': {'x': 2, 'y': 2}, 'costs': 2.8284, 'execution_mode': 'autonomous', 'entity': {'entity_type': 'victim', 'entity_super_type': 'object', 'id': 2, 'parent_id': 1}}]

Example: 'go to the box, but don't traverse'
Result: ERROR, There is no valid path to reach box 3


1. st scenario
# We are now in a new scenario:
Now apply the introduced concept on our new scenario. 
We are in a small building with 4 rooms. Room 1 is connected to room 2 by door 9 at 5.5 , 4. 
Room 3 is connected to room 1 with an open connection 0 at 9, 6.5. Room 3 is connected to room 4 with door 16 at 15.5, 7.
The center of room 1 is at 6, 9.
The center of room 2 is at 7, 3.
The center of room 3 is at 16, 6.
The center of room 4 is at 15.5, 10.5.

Door 1 is at 6.5, 13 in room 1. It's not known where it connects Room 1 to.
In room 1 are the following additional objects:
table 2 at 0.5, 10.5
table 3 at 0.5, 8
shelf 4 at 2, 4.5
container 5 at 12.5, 9
container 6 at 12.5, 10.5
container 7 at 12.5, 12.5
and shelf 8 at 3, 13.

In room 2 are the following additional objects:
cupboard 10 at 4.5, 3.5
cupboard 11 at 4.5, 2
cupboard 12 at 9.5, 2
cupboard 13 at 9.5, 3
table 14 at 7, 4
and table 15 at 8.5, 4

In room 3 are the following additional objects:
shelf 17 at 19, 4.5
container 18 at 21, 5.5
table 19 at 21.5, 6.5
and victim 20 at 19.5, 12.5

In room 4 are the following additional objects:
victim 21 at 14, 9
container 22 at 17.5, 9.5
container 23 at 17, 11
shelf 24 at 15.5, 12.5

To change the room, the robot must traverse either a door or an open connection.
The costs for traversing a door are 20 teleop, 30 assisted and 40 autonomous.
For open connections the traversal cost is 0 for all 3 modes, but autonomous traversal is preferred.
For all other actions the mode with the lowest cost is preferred.
The robot must first go to a door or connection before it can traverse it. Therefore before the robot can use the traverse action of a door or connection its position must be at the door or connection. Insert a goto action if needed.

The robot can go to each object of the same room. The cost is the euclidean distance to the object for all 3 modes. To go to a room means to visit its center position which is given in the scenario description.
  
if the robot is at a table, shelf, container or cupboard, it can inspect them with inspect or measure for a hazardous source with measure_hazard. The costs of inspect are 40 teleop, 35 autonomous and 30 assisted. The costs for measure_hazard are 40 teleop, 35 autonomous and 30 assisted.

tables and shelf can additionally be searched with search. The costs are 80 teleop, 70 autonomous and 60 assisted.
The robot can communicate with victims if he is at his position. The cost for this is 60 when teleoperated. This can only be done teleoperated.

Your aim is to give the set of actions with the lowest combined cost.

The robot starts in room 1 at 6.5, 12

Now perform the following task:
"""

#print(system_prompt)

#user_prompt = "go to the victim in room 4"

f = open("questions.txt")
questions = f.readlines()
questions = [question.strip() for question in questions]
f.close()

print(f"{len(questions)} questions loaded")
#print(questions)
  
# "min_p": 0.05,  not working on other machine??
options = {"temperature" : 0.0, "num_ctx": 8192, "top_k": 20, "num_predict": -1, "repeat_penalty": 1.0, "top_p": 0.1}


for i in range(0, len(questions)):
    f = open(f"question{i+1}", "w")
    print(questions[i])
    messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': questions[i]}]
    result = client.chat(model=model, messages=messages, options=options)
    print(result['message']['content'])
    f.write(questions[i])
    f.write("\n\n\n")
    f.write(result['message']['content'])
    f.close()
    print(f"saved question {i}")


#messages = [{'role': 'user', 'content': system_prompt + user_prompt}]

#result = client.generate(model=model, prompt=user_prompt,system=system_prompt,options=options)
#print(result['message']['content'])
