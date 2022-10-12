# Copyright 2022 Technion project
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
from unified_planning.test import TestCase, main
from unified_planning.test.examples.multi_agent import get_example_problems
from unified_planning.social_law.single_agent_projection import SingleAgentProjection
from unified_planning.model.multi_agent import *
from unified_planning.io import PDDLWriter

class TestProblem(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        self.problems = get_example_problems()

    def test_intersection_single_agent_projects(self):
        problem = self.problems["intersection"].problem

        sap = SingleAgentProjection(problem.agents[0])        
        result = sap.compile(problem)

        w = PDDLWriter(result.problem)

        f = open("domain.pddl", "w")
        f.write(w.get_domain())
        f.close()

        f = open("problem.pddl", "w")
        f.write(w.get_problem())
        f.close()

        