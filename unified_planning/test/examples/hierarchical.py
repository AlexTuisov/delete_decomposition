# Copyright 2021 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.model.htn import *
from collections import namedtuple

Example = namedtuple('Example', ['problem', 'plan'])

def get_example_problems():
    problems = {}

    Location = UserType("Location")
    loc = Fluent("loc", Location)

    move = Action("move", l1=Location, l2=Location)

    connected = Fluent("connected", l1=Location, l2=Location)
    go = Task("go", target=Location)

    go_direct = Method("go-direct", go, source=Location, target=Location)
    source = go_direct.parameter("source")
    target = go_direct.parameter("target")
    go_direct.add_precondition(Equals(loc, source))
    go_direct.add_precondition(connected(source, target))
    go_direct.add_subtask(move, source, target)
    print(go_direct)

    go_indirect = Method("go-indirect", go, source=Location, inter=Location, target=Location)
    source = go_indirect.parameter("source")
    inter = go_indirect.parameter("inter")
    target = go_indirect.parameter("target")
    go_indirect.add_precondition(Equals(loc, source))
    go_indirect.add_precondition(connected(source, inter))
    t1 = go_indirect.add_subtask(move, source, inter)
    t2 = go_indirect.add_subtask(go, target)
    go_indirect.set_ordered(t1, t2)
    print(go_indirect)



    # basic
    # x = Fluent('x')
    # a = InstantaneousAction('a')
    # a.add_precondition(Not(x))
    # a.add_effect(x, True)
    # problem = Problem('basic')
    # problem.add_fluent(x)
    # problem.add_action(a)
    # problem.set_initial_value(x, False)
    # problem.add_goal(x)
    # plan = up.plan.SequentialPlan([up.plan.ActionInstance(a)])
    # basic = Example(problem=problem, plan=plan)
    # problems['basic'] = basic

    return problems


if __name__ == "__main__":
    for name, problem in get_example_problems().items():
        print(f"======= {name} ======")
        print(str(problem.problem))
