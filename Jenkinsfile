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
        sh 'pytest --junitxml results_unittests.xml --cov=pyqmri --integration-cover test/Unittests'
      }
    }
    stage('Integrationtests') {
      steps {
        sh 'ipcluster start&'
        sh 'pytest --junitxml results_integrationtests.xml --cov=pyqmri --integration-cover test/Integrationtests'
        sh 'coverage xml'
        sh 'ipcluster stop&'
      }
    }
  }
  post {
      always {
          cobertura coberturaReportFile: 'coverage.xml'
          junit 'results*.xml'
          recordIssues enabledForFailure: true, tool: pyLint(pattern: 'pylint.log')
          cleanWs()
      }
  }
}
