# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stand alone file to convert XML based test results to HTML."""

import sys
import xml.etree.ElementTree as ET
import html


def generate_junit_html_report(xml_content):
  """
  Converts XML content to an HTML report.
  """
  try:
    root = ET.fromstring(xml_content)
  except ET.ParseError as e:
    return f'<html><body><h1>Error parsing XML</h1><p>{e}</p></body></html>'

  html_output = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Execution Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        h1, h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .summary { display: flex; justify-content: space-around; margin-bottom: 20px; }
        .summary-box {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            flex-grow: 1;
            margin: 0 10px;
        }
        .summary-box h3 { margin-top: 0; color: #495057; }
        .summary-box .count { font-size: 2em; font-weight: bold; }
        .tests-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .tests-table th, .tests-table td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        .tests-table th { background-color: #007bff; color: white; }
        .status-skipped { background-color: #fff3cd; color: #856404; }
        .status-passed { background-color: #d4edda; color: #155724; }
        .status-failed { background-color: #f8d7da; color: #721c24; }
        .status-errored { background-color: #f5c6cb; color: #721c24; font-weight: bold; }
        .message { white-space: pre-wrap; font-family: monospace; font-size: 0.9em; }
        .testsuite-header { margin-bottom: 15px; }
        .timestamp { font-size: 0.9em; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Execution Report</h1>
"""

  # Assuming <testsuites> can contain multiple <testsuite>
  for testsuite in root.findall('testsuite'):
    suite_name = testsuite.get('name', 'Unnamed Suite')
    total_tests = int(testsuite.get('tests', 0))
    failures = int(testsuite.get('failures', 0))
    errors = int(testsuite.get('errors', 0))
    skipped = int(testsuite.get('skipped', 0))
    time_taken = float(testsuite.get('time', 0.0))
    timestamp = testsuite.get('timestamp', 'N/A')
    hostname = testsuite.get('hostname', 'N/A')
    passed_tests = total_tests - (failures + errors + skipped)

    html_output += f'<h2>Test Suite: {html.escape(suite_name)}</h2>'
    html_output += f"<div class='timestamp'>Executed on: {html.escape(hostname)} at {html.escape(timestamp)}</div>"

    html_output += """
        <div class="summary">
            <div class="summary-box">
                <h3>Total Tests</h3>
                <div class="count" style="color: #007bff;">{total_tests}</div>
            </div>
            <div class="summary-box">
                <h3>Passed</h3>
                <div class="count" style="color: #28a745;">{passed_tests}</div>
            </div>
            <div class="summary-box">
                <h3>Failures</h3>
                <div class="count" style="color: #dc3545;">{failures}</div>
            </div>
            <div class="summary-box">
                <h3>Errors</h3>
                <div class="count" style="color: #c82333;">{errors}</div>
            </div>
            <div class="summary-box">
                <h3>Skipped</h3>
                <div class="count" style="color: #ffc107;">{skipped}</div>
            </div>
             <div class="summary-box">
                <h3>Time (s)</h3>
                <div class="count" style="color: #17a2b8;">{time_taken:.3f}</div>
            </div>
        </div>
        """.format(
        total_tests=total_tests,
        passed_tests=passed_tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        time_taken=time_taken,
    )

    html_output += """
        <table class="tests-table">
            <thead>
                <tr>
                    <th>Class Name</th>
                    <th>Test Name</th>
                    <th>Time (s)</th>
                    <th>Status</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
        """

    for testcase in testsuite.findall('testcase'):
      classname = testcase.get('classname', 'N/A')
      name = testcase.get('name', 'N/A')
      case_time = float(testcase.get('time', 0.0))
      status = 'Passed'
      status_class = 'status-passed'
      message_content = ''

      skipped_tag = testcase.find('skipped')
      failure_tag = testcase.find('failure')
      error_tag = testcase.find('error')

      if skipped_tag is not None:
        status = 'Skipped'
        status_class = 'status-skipped'
        message_content = skipped_tag.get('message', '')
        if skipped_tag.text:
          message_content += '<br>' + html.escape(skipped_tag.text.strip())
      elif failure_tag is not None:
        status = 'Failed'
        status_class = 'status-failed'
        message_content = failure_tag.get('message', '')
        if failure_tag.text:
          message_content += (
              '<br><pre>' + html.escape(failure_tag.text.strip()) + '</pre>'
          )
      elif error_tag is not None:
        status = 'Errored'
        status_class = 'status-errored'
        message_content = error_tag.get('message', '')
        if error_tag.text:
          message_content += (
              '<br><pre>' + html.escape(error_tag.text.strip()) + '</pre>'
          )

      html_output += f"""
                <tr>
                    <td>{html.escape(classname)}</td>
                    <td>{html.escape(name)}</td>
                    <td>{case_time:.3f}</td>
                    <td class="{status_class}">{status}</td>
                    <td class="message">{message_content if message_content else 'N/A'}</td>
                </tr>
            """

    html_output += """
            </tbody>
        </table>
        """

  html_output += """
    </div>
</body>
</html>
"""
  return html_output


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print(f'usage: {sys.argv[0]} <xml_input_file> <html_output_filename>')
    sys.exit()

  xmlfile = sys.argv[1]
  htmlfile = sys.argv[2]
  print(f'Converting XML test results in {xmlfile} to {htmlfile}')

  try:
    with open(xmlfile, 'r') as xmlin:
      htmlcode = generate_junit_html_report(xmlin.read())
      with open(htmlfile, 'w') as htmlout:
        htmlout.write(htmlcode)
  except NameError:
    print('Error processing files')
    sys.exit(1)

  print('Successfully wrote out Test results HTML file')
  sys.exit(0)
