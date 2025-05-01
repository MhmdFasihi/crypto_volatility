# Deployment Guide

## Local Deployment

### Development Environment

1. **Setup**
   ```bash
   git clone https://github.com/yourusername/crypto_volatility_analysis.git
   cd crypto_volatility_analysis
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration**
   - Copy `.env.example` to `.env`
   - Set environment variables
   - Adjust `src/config.py` as needed

3. **Run Application**
   ```bash
   streamlit run src/dashboard.py
   ```

### Production Environment

1. **System Requirements**
   - Ubuntu 20.04+ or similar
   - Python 3.8+
   - 8GB RAM minimum
   - 50GB storage

2. **Installation**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   ```

3. **Application Setup**
   ```bash
   mkdir -p /opt/crypto_volatility
   cd /opt/crypto_volatility
   git clone <repository-url> .
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Docker Deployment

### Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  crypto-volatility:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

### Build and Run

```bash
docker-compose build
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

1. **EC2 Setup**
   - Launch t3.medium instance
   - Configure security groups (port 8501)
   - Install Docker

2. **Deploy Application**
   ```bash
   ssh ec2-user@<instance-ip>
   git clone <repository-url>
   cd crypto_volatility_analysis
   docker-compose up -d
   ```

3. **Configure Load Balancer**
   - Create Application Load Balancer
   - Configure health checks
   - Set up SSL/TLS

### Google Cloud Platform

1. **Cloud Run Deployment**
   ```bash
   gcloud run deploy crypto-volatility \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

2. **Configure Resources**
   - Set memory limits: 2GB
   - Set CPU: 2 vCPU
   - Configure autoscaling

### Heroku Deployment

1. **Create Procfile**
   ```
   web: streamlit run src/dashboard.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**
   ```bash
   heroku create crypto-volatility-app
   git push heroku main
   heroku config:set PYTHON_VERSION=3.9
   ```

## Reverse Proxy Setup

### Nginx Configuration

Create `/etc/nginx/sites-available/crypto-volatility`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/crypto-volatility /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Process Management

### Systemd Service

Create `/etc/systemd/system/crypto-volatility.service`:
```ini
[Unit]
Description=Crypto Volatility Analysis Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/crypto_volatility
Environment="PATH=/opt/crypto_volatility/venv/bin"
ExecStart=/opt/crypto_volatility/venv/bin/streamlit run src/dashboard.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable crypto-volatility
sudo systemctl start crypto-volatility
```

### PM2 Management

```bash
npm install -g pm2
pm2 start "streamlit run src/dashboard.py" --name crypto-volatility
pm2 save
pm2 startup
```

## Monitoring

### Health Checks

Create `healthcheck.py`:
```python
import requests
import sys

try:
    response = requests.get('http://localhost:8501/_stcore/health')
    if response.status_code == 200:
        sys.exit(0)
    else:
        sys.exit(1)
except:
    sys.exit(1)
```

### Logging Setup

Configure logging in `src/config.py`:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Monitoring Tools

1. **Prometheus Integration**
   ```python
   from prometheus_client import Counter, Histogram
   
   REQUEST_COUNT = Counter('app_requests_total', 'Total requests')
   REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')
   ```

2. **Grafana Dashboard**
   - Import dashboard templates
   - Configure data sources
   - Set up alerts

## Security

### SSL/TLS Setup

1. **Let's Encrypt**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

2. **Auto-renewal**
   ```bash
   sudo systemctl enable certbot.timer
   sudo systemctl start certbot.timer
   ```

### Authentication

Add basic authentication:
```python
# In src/dashboard.py
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')
```

## Backup Strategy

### Automated Backups

Create backup script:
```bash
#!/bin/bash
BACKUP_DIR="/backups/crypto-volatility"
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz ./models/

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz ./data/

# Backup configs
tar -czf $BACKUP_DIR/config_$DATE.tar.gz ./src/config.py

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Schedule with cron:
```bash
0 2 * * * /path/to/backup.sh
```

## Scaling

### Horizontal Scaling

1. **Load Balancer Setup**
   - Configure sticky sessions
   - Health check endpoints
   - SSL termination

2. **Session Management**
   - Use Redis for sessions
   - Configure session affinity

### Performance Optimization

1. **Caching**
   ```python
   @st.cache_data(ttl=3600)
   def load_data(ticker, start_date, end_date):
       return get_data(ticker, start_date, end_date)
   ```

2. **Database Optimization**
   - Use connection pooling
   - Implement query caching
   - Optimize indexes

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -i :8501
   kill -9 <PID>
   ```

2. **Memory Issues**
   - Increase swap space
   - Optimize data loading
   - Use chunked processing

3. **Connection Errors**
   - Check firewall settings
   - Verify network configuration
   - Test API endpoints

### Logs Location

- Application logs: `/var/log/crypto-volatility/`
- Nginx logs: `/var/log/nginx/`
- System logs: `/var/log/syslog`

## Maintenance

### Regular Tasks

1. **Update Dependencies**
   ```bash
   pip list --outdated
   pip install --upgrade <package>
   ```

2. **Clean Old Data**
   ```bash
   find ./data -name "*.csv" -mtime +30 -delete
   ```

3. **Rotate Logs**
   ```bash
   logrotate /etc/logrotate.d/crypto-volatility
   ```

### Health Monitoring

```bash
#!/bin/bash
# health_check.sh

if curl -s http://localhost:8501/_stcore/health | grep -q "ok"; then
    echo "Application is healthy"
else
    echo "Application is down"
    systemctl restart crypto-volatility
fi
```

## Version Updates

1. **Zero-downtime Deployment**
   ```bash
   # Blue-green deployment
   docker-compose -f docker-compose.blue.yml up -d
   # Test new version
   # Switch traffic
   docker-compose -f docker-compose.green.yml down
   docker-compose -f docker-compose.blue.yml down
   ```

2. **Rolling Updates**
   ```bash
   # Update one instance at a time
   kubectl rollout restart deployment crypto-volatility
   kubectl rollout status deployment crypto-volatility
   ```

3. **Database Migrations**
   ```bash
   # Backup before migration
   python manage.py backup_db
   # Run migrations
   python manage.py migrate
   # Verify
   python manage.py check_db
   ```

## Disaster Recovery

### Backup Verification

```bash
#!/bin/bash
# verify_backup.sh

BACKUP_FILE=$1
TEMP_DIR=$(mktemp -d)

tar -xzf $BACKUP_FILE -C $TEMP_DIR
if [ $? -eq 0 ]; then
    echo "Backup verified successfully"
    rm -rf $TEMP_DIR
    exit 0
else
    echo "Backup verification failed"
    exit 1
fi
```

### Recovery Procedures

1. **Full System Recovery**
   ```bash
   # Stop services
   systemctl stop crypto-volatility
   
   # Restore from backup
   tar -xzf /backups/latest/models.tar.gz -C /opt/crypto_volatility/
   tar -xzf /backups/latest/data.tar.gz -C /opt/crypto_volatility/
   
   # Start services
   systemctl start crypto-volatility
   ```

2. **Database Recovery**
   ```bash
   # Restore database
   psql -U postgres crypto_volatility < backup.sql
   
   # Verify integrity
   python manage.py check_db
   ```

## Performance Tuning

### Application Optimization

1. **Memory Settings**
   ```python
   # In src/config.py
   import gc
   
   # Force garbage collection
   gc.enable()
   gc.set_threshold(700, 10, 10)
   ```

2. **Streamlit Configuration**
   ```toml
   # .streamlit/config.toml
   [server]
   maxUploadSize = 200
   maxMessageSize = 200
   
   [browser]
   serverAddress = "0.0.0.0"
   serverPort = 8501
   
   [runner]
   fastReruns = true
   ```

### Database Optimization

1. **Connection Pooling**
   ```python
   from sqlalchemy import create_engine
   from sqlalchemy.pool import QueuePool
   
   engine = create_engine(
       'postgresql://user:pass@localhost/db',
       poolclass=QueuePool,
       pool_size=20,
       max_overflow=0
   )
   ```

2. **Query Optimization**
   ```sql
   -- Create indexes
   CREATE INDEX idx_volatility_date ON volatility_data(date);
   CREATE INDEX idx_volatility_ticker ON volatility_data(ticker);
   ```

## CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /opt/crypto_volatility
          git pull
          docker-compose down
          docker-compose up -d --build
```

### Jenkins Pipeline

Create `Jenkinsfile`:
```groovy
pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t crypto-volatility .'
            }
        }
        
        stage('Test') {
            steps {
                sh 'docker run crypto-volatility pytest tests/'
            }
        }
        
        stage('Deploy') {
            steps {
                sh '''
                    docker-compose down
                    docker-compose up -d
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
```

## Environment Variables

### Production Settings

Create `.env.production`:
```bash
# API Keys
YFINANCE_API_KEY=your_key_here
DERIBIT_API_KEY=your_key_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crypto_volatility
DB_USER=app_user
DB_PASSWORD=secure_password

# Application
DEBUG=False
LOG_LEVEL=INFO
WORKERS=4

# Security
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=your-domain.com
```

### Docker Environment

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build: .
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data:rw
      - ./models:/app/models:ro
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Security Checklist

### Pre-deployment

- [ ] Remove debug mode
- [ ] Update all dependencies
- [ ] Scan for vulnerabilities
- [ ] Configure firewall
- [ ] Set up SSL/TLS
- [ ] Enable authentication
- [ ] Secure API keys
- [ ] Configure CORS
- [ ] Set up rate limiting
- [ ] Enable logging

### Post-deployment

- [ ] Verify SSL certificate
- [ ] Test authentication
- [ ] Check error pages
- [ ] Monitor logs
- [ ] Set up alerts
- [ ] Test backups
- [ ] Verify monitoring
- [ ] Document access

## Maintenance Schedule

### Daily Tasks

- Monitor system health
- Check error logs
- Verify data updates
- Review security alerts

### Weekly Tasks

- Update models
- Clean old data
- Test backups
- Review performance

### Monthly Tasks

- Security audit
- Dependency updates
- Performance review
- Capacity planning

## Support and Troubleshooting

### Contact Information

- Technical Support: support@example.com
- Emergency: +1-XXX-XXX-XXXX
- Documentation: https://docs.example.com

### Emergency Procedures

1. **System Down**
   ```bash
   # Quick restart
   systemctl restart crypto-volatility
   
   # Check logs
   tail -f /var/log/crypto-volatility/error.log
   ```

2. **Data Corruption**
   ```bash
   # Stop application
   systemctl stop crypto-volatility
   
   # Restore from backup
   ./scripts/restore_backup.sh
   
   # Verify data
   python scripts/verify_data.py
   ```

3. **Security Breach**
   ```bash
   # Isolate system
   iptables -A INPUT -j DROP
   
   # Preserve evidence
   tar -czf incident_$(date +%Y%m%d).tar.gz /var/log/
   
   # Notify security team
   ./scripts/security_alert.sh
   ```

## Conclusion

This deployment guide covers:
- Local and cloud deployment options
- Docker and container orchestration
- Security and monitoring setup
- Backup and recovery procedures
- CI/CD pipeline configuration
- Maintenance and troubleshooting

Follow these guidelines to ensure a secure, scalable, and maintainable deployment of the Crypto Volatility Analysis framework.