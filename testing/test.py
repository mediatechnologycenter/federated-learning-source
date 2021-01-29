#   Copyright 2021, ETH Zurich, Media Technology Center
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import test_class
import logging
import os


#todo check add noise for >1000 categories
logging.basicConfig(
    level=int(os.getenv('LOGGING_LEVEL', 0)),
    format=f"%(asctime)s [%(processName)-12.12s] [%(levelname)-5.5s] [test] [%(filename)s / %(funcName)s / %(lineno)d] %(message)s")

test_cases = [os.getenv('test')] if os.getenv('test', None) else None
test_cases = sorted([test_case for test_case in os.listdir('testing/test_wrapper') if 'test' in test_case])[14:16]

print(test_cases)

Test = test_class.Tests(start_servers=False, clear_logs=True)

Test.clear_db()

test_result = Test.run_all_tests(test_cases=test_cases)
logging.info(test_result)
if not all(test_result.values()):
    logging.error("TESTS FAILED")
Test.kill_servers()
