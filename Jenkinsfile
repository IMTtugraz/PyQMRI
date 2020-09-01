pipeline { 
  agent {     
    dockerfile {
      filename 'Dockerfile'
      args '--gpus all -u root'
      }
  }
  stages {
    stage('Checkout') {
      steps {
         sh 'ls -la test'
        } 
    }
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
    stage('Testing') {
      steps {
        sh 'pytest --junitxml results.xml --cov=pyqmri --integration-cover test/'
        sh 'coverage xml'
      }
    }
  }
  post {
      always {
          cobertura coberturaReportFile: 'coverage.xml'
          junit 'results.xml'
          recordIssues enabledForFailure: true, tool: pyLint(pattern: 'pylint.log')
      }
  }
}
