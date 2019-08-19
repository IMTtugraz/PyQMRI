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
        sh 'pytest --junitxml results.xml --cov=pyqmri test/'
        sh 'coverage xml'
      }
    }
    stage('Cleaning Up') {
      steps {
        sh 'rm -r ./test/__pycache__'
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
