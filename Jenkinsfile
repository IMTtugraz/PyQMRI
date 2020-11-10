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
        sh 'pytest --junitxml results_unittests_LinOp.xml --cov=pyqmri test/unittests/test_LinearDataOperator'
        sh 'coverage xml -o coverage_unittest_LinOp.xml'
        sh 'pytest --junitxml results_unittests_grad.xml --cov=pyqmri test/unittests/test_gradient'
        sh 'coverage xml -o coverage_unittest_grad.xml'
        sh 'pytest --junitxml results_unittests_symgrad.xml --cov=pyqmri test/unittests/test_symmetrized_gradient'
        sh 'coverage xml -o coverage_unittest_symgrad.xml'
      }
    }
    stage('Integrationtests') {
      steps {
        sh 'ipcluster start&'
        sh 'pytest --junitxml results_integrationtests.xml --cov=pyqmri --integration-cover test/integrationtests/'
        sh 'coverage xml -o coverage_integrationtest.xml'
        sh 'ipcluster stop&'
      }
    }
  }
  post {
      always {
          cobertura coberturaReportFile: 'coverage_unittest_LinOp.xml, coverage_unittest_grad.xml, coverage_unittest_symgrad.xml, coverage_integrationtest.xml', enableNewApi: true
          junit 'results*.xml'
          recordIssues enabledForFailure: true, tool: pyLint(pattern: 'pylint.log')
          cleanWs()
      }
  }
}
