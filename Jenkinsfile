pipeline { 
  agent {     
    dockerfile {
      filename 'Dockerfile'
      args '--gpus all -u root'
      }
  }
  stages {
    stage('Build') {
      steps {
        sh 'pip3 install -r requirements.txt'
        sh 'pip3 install -e .'
      }
    }
    stage('Pylint') {
      steps {
        sh 'pylint -ry --output-format=parseable --exit-zero ./pyqmri > pylint.log'
      }
    }
    stage('Unittests') {
      steps {
        sh 'pytest --junitxml results_unittests_LinOp.xml --cov=pyqmri test/unittests/test_LinearDataOperator.py'
        sh 'coverage xml -o coverage_unittest_LinOp.xml'
        sh 'pytest --junitxml results_unittests_grad.xml --cov=pyqmri test/unittests/test_gradient.py'
        sh 'coverage xml -o coverage_unittest_grad.xml'
        sh 'pytest --junitxml results_unittests_symgrad.xml --cov=pyqmri test/unittests/test_symmetrized_gradient.py'
        sh 'coverage xml -o coverage_unittest_symgrad.xml'
      }
    }
    stage('Integrationtests') {
      steps {
        sh 'ipcluster start&'
        sh 'pytest --junitxml results_integrationtests_single_slice.xml --cov=pyqmri --integration-cover test/integrationtests/test_integration_test_single_slice.py'
        sh 'coverage xml -o coverage_integrationtest_single_slice.xml'
        sh 'pytest --junitxml results_integrationtests_multi_slice.xml --cov=pyqmri --integration-cover test/integrationtests/test_integration_test_multi_slice.py'
        sh 'coverage xml -o coverage_integrationtest_multi_slice.xml'
        sh 'ipcluster stop&'
      }
    }
  }
  post {
      always {
          cobertura coberturaReportFile: 'coverage_unittest_LinOp.xml, coverage_unittest_grad.xml, coverage_unittest_symgrad.xml, coverage_integrationtest_single_slice.xml, coverage_integrationtest_multi_slice.xml', enableNewApi: true
          junit 'results*.xml'
          recordIssues enabledForFailure: true, tool: pyLint(pattern: 'pylint.log')
          step([$class: 'GitHubCommitStatusSetter', statusBackrefSource: [$class: 'ManuallyEnteredBackrefSource', backref: 'https://00c8da69b5ae.ngrok.io:8090/job/PyQMRI_public/job/JOSS_pub/42/display/redirect']])
          cleanWs()
      }
  }
}
