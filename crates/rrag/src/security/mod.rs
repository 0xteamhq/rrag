//! # RRAG Security Framework
//! 
//! Comprehensive security system for RAG applications with enterprise-grade features:
//! - Multi-factor authentication (API keys, JWT, OAuth2, TOTP, WebAuthn)
//! - Advanced rate limiting with multiple strategies
//! - Input validation and sanitization
//! - Security headers and CORS management
//! - Request/response encryption
//! - Audit logging and monitoring
//! - API abuse detection and prevention
//! - Resource quotas and throttling
//! - Secure configuration management
//! - Vulnerability assessment tools

pub mod auth;
pub mod rate_limit;
pub mod validation;
pub mod encryption;
pub mod audit;
pub mod abuse_detection;
pub mod quotas;
pub mod config;
pub mod headers;
pub mod vulnerability;
pub mod monitoring;
pub mod middleware;
pub mod session;

// Re-exports for convenient access
pub use auth::{
    AuthenticationService, AuthorizationService, ApiKeyAuth, JwtAuth, 
    OAuth2Auth, TotpAuth, WebAuthnAuth, AuthProvider, AuthConfig,
    Credentials, AuthResult, Permission, Role, User, Session,
    AuthenticationError, AuthorizationError
};

pub use rate_limit::{
    RateLimiter, RateLimitStrategy, RateLimitConfig, RateLimitResult,
    TokenBucket, SlidingWindow, FixedWindow, LeakyBucket,
    RateLimitStore, InMemoryRateLimitStore, RedisRateLimitStore,
    RateLimitKey, RateLimitError, RateLimitInfo, RateLimitQuota
};

pub use validation::{
    InputValidator, ValidationRule, ValidationResult, ValidationError,
    Sanitizer, SanitizationRule, SanitizationResult,
    RequestValidator, ResponseValidator, SchemaValidator,
    XssProtection, SqlInjectionProtection, PathTraversalProtection,
    ContentSecurityPolicy, ValidationConfig
};

pub use encryption::{
    EncryptionService, EncryptionConfig, EncryptionAlgorithm,
    RequestEncryption, ResponseEncryption, DataProtection,
    KeyManager, EncryptionKey, DecryptionKey, CryptoProvider,
    Aes256Gcm, ChaCha20Poly1305, EncryptionError, DecryptionError
};

pub use audit::{
    AuditLogger, AuditEvent, AuditLevel, AuditConfig, AuditResult,
    SecurityEvent, AuthenticationEvent, AuthorizationEvent,
    RateLimitEvent, ValidationEvent, AbuseDetectionEvent,
    AuditStorage, FileAuditStorage, DatabaseAuditStorage,
    AuditQuery, AuditReport, AuditAnalytics
};

pub use abuse_detection::{
    AbuseDetector, AbuseDetectionConfig, AbusePattern, AbuseRule,
    AnomalyDetector, BehaviorAnalyzer, ThreatIntelligence,
    IpReputation, GeoLocation, DeviceFingerprinting,
    BotDetection, CaptchaChallenge, AbuseResponse, AbuseMetrics
};

pub use quotas::{
    QuotaManager, QuotaConfig, QuotaLimit, QuotaUsage, QuotaResult,
    ResourceQuota, UserQuota, ApiQuota, TimeWindowQuota,
    QuotaEnforcer, QuotaStorage, QuotaMetrics, ThrottleManager,
    QueueManager, BackpressureController
};

pub use config::{
    SecurityConfig, SecurityConfigBuilder, ConfigValidator,
    SecureConfigManager, ConfigEncryption, EnvironmentConfig,
    VaultIntegration, SecretsManager, ConfigAudit, ConfigVersioning
};

pub use headers::{
    SecurityHeaders, SecurityHeadersConfig, CorsConfig, CorsPolicy,
    ContentSecurityPolicyBuilder, HstsConfig, XFrameOptions,
    XContentTypeOptions, ReferrerPolicy, PermissionsPolicy,
    CrossOriginPolicy, SecurityHeadersMiddleware
};

pub use vulnerability::{
    VulnerabilityScanner, SecurityAssessment, PenetrationTesting,
    DependencyScanner, CodeAnalyzer, ConfigurationReview,
    ThreatModeling, RiskAssessment, ComplianceChecker,
    SecurityReport, VulnerabilityReport, RemediationPlan
};

pub use monitoring::{
    SecurityMonitor, SecurityMetrics, SecurityDashboard,
    ThreatDetection, IncidentResponse, AlertManager,
    SecurityAnalytics, RealTimeMonitoring, ComplianceMonitoring,
    SecurityKpi, ThreatIntelligenceFeed, SiemIntegration
};

pub use middleware::{
    SecurityMiddleware, AuthMiddleware, RateLimitMiddleware,
    ValidationMiddleware, AuditMiddleware, EncryptionMiddleware,
    HeadersMiddleware, SecurityStack, MiddlewareChain
};

pub use session::{
    SessionManager, SessionStore, SessionConfig, SessionSecurity,
    SessionToken, SessionData, SessionValidator, SessionCleaner,
    DistributedSessionStore, SessionReplication, SessionBackup
};

use crate::{RragError, RragResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Main security service that orchestrates all security components
#[derive(Clone)]
pub struct SecurityService {
    auth_service: Arc<AuthenticationService>,
    authz_service: Arc<AuthorizationService>,
    rate_limiter: Arc<dyn RateLimiter>,
    input_validator: Arc<InputValidator>,
    encryption_service: Arc<EncryptionService>,
    audit_logger: Arc<AuditLogger>,
    abuse_detector: Arc<AbuseDetector>,
    quota_manager: Arc<QuotaManager>,
    security_config: Arc<SecurityConfig>,
    session_manager: Arc<SessionManager>,
    monitor: Arc<SecurityMonitor>,
    config: SecurityServiceConfig,
}

/// Configuration for the security service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityServiceConfig {
    /// Enable authentication
    pub enable_auth: bool,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Enable input validation
    pub enable_validation: bool,
    /// Enable request/response encryption
    pub enable_encryption: bool,
    /// Enable audit logging
    pub enable_audit: bool,
    /// Enable abuse detection
    pub enable_abuse_detection: bool,
    /// Enable resource quotas
    pub enable_quotas: bool,
    /// Enable session management
    pub enable_sessions: bool,
    /// Enable real-time monitoring
    pub enable_monitoring: bool,
    /// Security level (1-5, where 5 is maximum security)
    pub security_level: u8,
    /// Enable FIPS compliance mode
    pub fips_mode: bool,
}

impl Default for SecurityServiceConfig {
    fn default() -> Self {
        Self {
            enable_auth: true,
            enable_rate_limiting: true,
            enable_validation: true,
            enable_encryption: false,
            enable_audit: true,
            enable_abuse_detection: true,
            enable_quotas: true,
            enable_sessions: true,
            enable_monitoring: true,
            security_level: 3,
            fips_mode: false,
        }
    }
}

/// Security context for a request/operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Request ID for tracking
    pub request_id: Uuid,
    /// Authenticated user information
    pub user: Option<User>,
    /// Session information
    pub session: Option<Session>,
    /// IP address of the client
    pub client_ip: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Request timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Security labels/tags
    pub security_labels: HashMap<String, String>,
    /// Risk score (0.0 - 1.0)
    pub risk_score: f32,
    /// Authentication method used
    pub auth_method: Option<String>,
    /// Rate limit information
    pub rate_limit_info: Option<RateLimitInfo>,
    /// Validation results
    pub validation_results: Vec<ValidationResult>,
}

impl SecurityContext {
    /// Create a new security context
    pub fn new() -> Self {
        Self {
            request_id: Uuid::new_v4(),
            user: None,
            session: None,
            client_ip: None,
            user_agent: None,
            timestamp: chrono::Utc::now(),
            security_labels: HashMap::new(),
            risk_score: 0.0,
            auth_method: None,
            rate_limit_info: None,
            validation_results: Vec::new(),
        }
    }

    /// Check if the context represents an authenticated user
    pub fn is_authenticated(&self) -> bool {
        self.user.is_some()
    }

    /// Check if the context has a valid session
    pub fn has_valid_session(&self) -> bool {
        self.session.as_ref().map_or(false, |s| !s.is_expired())
    }

    /// Get the user ID if authenticated
    pub fn user_id(&self) -> Option<&str> {
        self.user.as_ref().map(|u| u.id.as_str())
    }

    /// Add a security label
    pub fn add_label(&mut self, key: String, value: String) {
        self.security_labels.insert(key, value);
    }

    /// Update risk score
    pub fn update_risk_score(&mut self, score: f32) {
        self.risk_score = score.clamp(0.0, 1.0);
    }

    /// Check if risk score exceeds threshold
    pub fn is_high_risk(&self, threshold: f32) -> bool {
        self.risk_score > threshold
    }
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Security decision result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityDecision {
    Allow,
    Deny { reason: String },
    Challenge { challenge_type: String },
    Throttle { delay_ms: u64 },
}

impl SecurityDecision {
    /// Check if the decision allows the operation
    pub fn is_allowed(&self) -> bool {
        matches!(self, SecurityDecision::Allow)
    }

    /// Check if the decision denies the operation
    pub fn is_denied(&self) -> bool {
        matches!(self, SecurityDecision::Deny { .. })
    }

    /// Check if the decision requires a challenge
    pub fn requires_challenge(&self) -> bool {
        matches!(self, SecurityDecision::Challenge { .. })
    }

    /// Check if the decision requires throttling
    pub fn requires_throttle(&self) -> bool {
        matches!(self, SecurityDecision::Throttle { .. })
    }
}

/// Security operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityOperation {
    Authentication,
    Authorization,
    DataAccess,
    DocumentIngestion,
    QueryExecution,
    AdminOperation,
    ConfigurationChange,
    SystemMonitoring,
}

impl std::fmt::Display for SecurityOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Authentication => write!(f, "authentication"),
            Self::Authorization => write!(f, "authorization"),
            Self::DataAccess => write!(f, "data_access"),
            Self::DocumentIngestion => write!(f, "document_ingestion"),
            Self::QueryExecution => write!(f, "query_execution"),
            Self::AdminOperation => write!(f, "admin_operation"),
            Self::ConfigurationChange => write!(f, "configuration_change"),
            Self::SystemMonitoring => write!(f, "system_monitoring"),
        }
    }
}

/// Security service implementation
impl SecurityService {
    /// Create a new security service builder
    pub fn builder() -> SecurityServiceBuilder {
        SecurityServiceBuilder::new()
    }

    /// Evaluate security for an operation
    pub async fn evaluate_security(
        &self,
        operation: SecurityOperation,
        context: &mut SecurityContext,
        data: Option<&[u8]>,
    ) -> RragResult<SecurityDecision> {
        // Record the security evaluation attempt
        self.audit_logger.log_security_evaluation(operation.clone(), context).await?;

        // Step 1: Authentication check
        if self.config.enable_auth && !context.is_authenticated() {
            let auth_required = self.requires_authentication(&operation)?;
            if auth_required {
                return Ok(SecurityDecision::Deny { 
                    reason: "Authentication required".to_string() 
                });
            }
        }

        // Step 2: Rate limiting
        if self.config.enable_rate_limiting {
            let rate_limit_result = self.check_rate_limits(context).await?;
            if !rate_limit_result.allowed {
                return Ok(SecurityDecision::Throttle { 
                    delay_ms: rate_limit_result.retry_after_ms.unwrap_or(1000) 
                });
            }
            context.rate_limit_info = Some(rate_limit_result.info);
        }

        // Step 3: Authorization check
        if self.config.enable_auth && context.is_authenticated() {
            let authz_result = self.authz_service.authorize(
                context.user.as_ref().unwrap(),
                &operation,
                None
            ).await?;

            if !authz_result.allowed {
                return Ok(SecurityDecision::Deny { 
                    reason: authz_result.reason.unwrap_or_else(|| "Access denied".to_string()) 
                });
            }
        }

        // Step 4: Input validation
        if self.config.enable_validation && data.is_some() {
            let validation_result = self.input_validator.validate_bytes(data.unwrap()).await?;
            context.validation_results.push(validation_result.clone());
            
            if !validation_result.is_valid {
                return Ok(SecurityDecision::Deny { 
                    reason: format!("Validation failed: {}", validation_result.error_message.unwrap_or_default())
                });
            }
        }

        // Step 5: Abuse detection
        if self.config.enable_abuse_detection {
            let abuse_result = self.abuse_detector.analyze_context(context).await?;
            if abuse_result.is_suspicious {
                context.update_risk_score(abuse_result.risk_score);
                
                if abuse_result.requires_challenge {
                    return Ok(SecurityDecision::Challenge { 
                        challenge_type: abuse_result.challenge_type.unwrap_or_else(|| "captcha".to_string())
                    });
                }
            }
        }

        // Step 6: Resource quotas
        if self.config.enable_quotas && context.is_authenticated() {
            let quota_result = self.quota_manager.check_quota(
                context.user_id().unwrap(),
                &operation,
            ).await?;

            if !quota_result.allowed {
                return Ok(SecurityDecision::Deny { 
                    reason: "Quota exceeded".to_string() 
                });
            }
        }

        // Step 7: Real-time monitoring and threat detection
        if self.config.enable_monitoring {
            self.monitor.record_operation(&operation, context).await?;
            
            let threat_result = self.monitor.detect_threats(context).await?;
            if threat_result.threat_detected {
                return Ok(SecurityDecision::Deny { 
                    reason: format!("Security threat detected: {}", threat_result.threat_type)
                });
            }
        }

        // All checks passed
        Ok(SecurityDecision::Allow)
    }

    /// Authenticate a user with credentials
    pub async fn authenticate(
        &self,
        credentials: Credentials,
        context: &mut SecurityContext,
    ) -> RragResult<AuthResult> {
        if !self.config.enable_auth {
            return Ok(AuthResult::success(User::anonymous()));
        }

        let auth_result = self.auth_service.authenticate(credentials).await?;
        
        if auth_result.success {
            context.user = auth_result.user.clone();
            context.auth_method = Some(auth_result.method.clone());
            
            // Create session if enabled
            if self.config.enable_sessions {
                let session = self.session_manager.create_session(
                    auth_result.user.as_ref().unwrap(),
                    context.client_ip.clone(),
                ).await?;
                context.session = Some(session);
            }
        }

        // Log authentication event
        self.audit_logger.log_authentication_event(&auth_result, context).await?;

        Ok(auth_result)
    }

    /// Check rate limits for the current context
    async fn check_rate_limits(&self, context: &SecurityContext) -> RragResult<RateLimitResult> {
        let key = self.build_rate_limit_key(context)?;
        self.rate_limiter.check_rate_limit(key).await
    }

    /// Build rate limit key from context
    fn build_rate_limit_key(&self, context: &SecurityContext) -> RragResult<RateLimitKey> {
        let mut key_parts = Vec::new();

        // Add user ID if authenticated
        if let Some(user_id) = context.user_id() {
            key_parts.push(format!("user:{}", user_id));
        }

        // Add IP address if available
        if let Some(ip) = &context.client_ip {
            key_parts.push(format!("ip:{}", ip));
        }

        // Add user agent hash if available
        if let Some(ua) = &context.user_agent {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(ua.as_bytes());
            let hash = format!("{:x}", hasher.finalize());
            key_parts.push(format!("ua:{}", &hash[..8]));
        }

        if key_parts.is_empty() {
            key_parts.push("anonymous".to_string());
        }

        Ok(RateLimitKey::new(key_parts.join(";")))
    }

    /// Check if operation requires authentication
    fn requires_authentication(&self, operation: &SecurityOperation) -> RragResult<bool> {
        match operation {
            SecurityOperation::Authentication => Ok(false),
            SecurityOperation::Authorization => Ok(true),
            SecurityOperation::DataAccess => Ok(true),
            SecurityOperation::DocumentIngestion => Ok(true),
            SecurityOperation::QueryExecution => Ok(false), // May be public
            SecurityOperation::AdminOperation => Ok(true),
            SecurityOperation::ConfigurationChange => Ok(true),
            SecurityOperation::SystemMonitoring => Ok(true),
        }
    }

    /// Get security metrics
    pub async fn get_security_metrics(&self) -> RragResult<SecurityMetrics> {
        self.monitor.get_metrics().await
    }

    /// Perform security health check
    pub async fn health_check(&self) -> RragResult<SecurityHealthReport> {
        let mut report = SecurityHealthReport::new();

        // Check authentication service
        if self.config.enable_auth {
            let auth_health = self.auth_service.health_check().await?;
            report.add_component_health("authentication", auth_health);
        }

        // Check rate limiter
        if self.config.enable_rate_limiting {
            let rate_limit_health = self.rate_limiter.health_check().await?;
            report.add_component_health("rate_limiting", rate_limit_health);
        }

        // Check other components...
        report.overall_health = report.calculate_overall_health();

        Ok(report)
    }
}

/// Security service builder
pub struct SecurityServiceBuilder {
    auth_service: Option<Arc<AuthenticationService>>,
    authz_service: Option<Arc<AuthorizationService>>,
    rate_limiter: Option<Arc<dyn RateLimiter>>,
    input_validator: Option<Arc<InputValidator>>,
    encryption_service: Option<Arc<EncryptionService>>,
    audit_logger: Option<Arc<AuditLogger>>,
    abuse_detector: Option<Arc<AbuseDetector>>,
    quota_manager: Option<Arc<QuotaManager>>,
    security_config: Option<Arc<SecurityConfig>>,
    session_manager: Option<Arc<SessionManager>>,
    monitor: Option<Arc<SecurityMonitor>>,
    config: SecurityServiceConfig,
}

impl SecurityServiceBuilder {
    pub fn new() -> Self {
        Self {
            auth_service: None,
            authz_service: None,
            rate_limiter: None,
            input_validator: None,
            encryption_service: None,
            audit_logger: None,
            abuse_detector: None,
            quota_manager: None,
            security_config: None,
            session_manager: None,
            monitor: None,
            config: SecurityServiceConfig::default(),
        }
    }

    pub fn with_authentication(mut self, auth_service: Arc<AuthenticationService>) -> Self {
        self.auth_service = Some(auth_service);
        self
    }

    pub fn with_authorization(mut self, authz_service: Arc<AuthorizationService>) -> Self {
        self.authz_service = Some(authz_service);
        self
    }

    pub fn with_rate_limiting(mut self, rate_limiter: Arc<dyn RateLimiter>) -> Self {
        self.rate_limiter = Some(rate_limiter);
        self
    }

    pub fn with_config(mut self, config: SecurityServiceConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn build(self) -> RragResult<SecurityService> {
        // Create default components if not provided
        let auth_service = self.auth_service.unwrap_or_else(|| {
            Arc::new(AuthenticationService::new(AuthConfig::default()))
        });

        let authz_service = self.authz_service.unwrap_or_else(|| {
            Arc::new(AuthorizationService::new())
        });

        let rate_limiter = self.rate_limiter.unwrap_or_else(|| {
            Arc::new(InMemoryRateLimitStore::new())
        });

        let input_validator = self.input_validator.unwrap_or_else(|| {
            Arc::new(InputValidator::new(ValidationConfig::default()))
        });

        let encryption_service = self.encryption_service.unwrap_or_else(|| {
            Arc::new(EncryptionService::new(EncryptionConfig::default()))
        });

        let audit_logger = self.audit_logger.unwrap_or_else(|| {
            Arc::new(AuditLogger::new(AuditConfig::default()))
        });

        let abuse_detector = self.abuse_detector.unwrap_or_else(|| {
            Arc::new(AbuseDetector::new(AbuseDetectionConfig::default()))
        });

        let quota_manager = self.quota_manager.unwrap_or_else(|| {
            Arc::new(QuotaManager::new(QuotaConfig::default()))
        });

        let security_config = self.security_config.unwrap_or_else(|| {
            Arc::new(SecurityConfig::default())
        });

        let session_manager = self.session_manager.unwrap_or_else(|| {
            Arc::new(SessionManager::new(SessionConfig::default()))
        });

        let monitor = self.monitor.unwrap_or_else(|| {
            Arc::new(SecurityMonitor::new())
        });

        Ok(SecurityService {
            auth_service,
            authz_service,
            rate_limiter,
            input_validator,
            encryption_service,
            audit_logger,
            abuse_detector,
            quota_manager,
            security_config,
            session_manager,
            monitor,
            config: self.config,
        })
    }
}

/// Security health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityHealthReport {
    pub overall_health: HealthStatus,
    pub component_health: HashMap<String, HealthStatus>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub issues: Vec<SecurityIssue>,
}

impl SecurityHealthReport {
    pub fn new() -> Self {
        Self {
            overall_health: HealthStatus::Unknown,
            component_health: HashMap::new(),
            timestamp: chrono::Utc::now(),
            issues: Vec::new(),
        }
    }

    pub fn add_component_health(&mut self, component: &str, health: HealthStatus) {
        self.component_health.insert(component.to_string(), health);
    }

    pub fn calculate_overall_health(&self) -> HealthStatus {
        if self.component_health.is_empty() {
            return HealthStatus::Unknown;
        }

        let has_critical = self.component_health.values().any(|h| *h == HealthStatus::Critical);
        let has_degraded = self.component_health.values().any(|h| *h == HealthStatus::Degraded);

        if has_critical {
            HealthStatus::Critical
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
}

impl Default for SecurityHealthReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    Unknown,
}

/// Security issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub severity: IssueSeverity,
    pub component: String,
    pub description: String,
    pub recommendation: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Issue severity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_context_creation() {
        let context = SecurityContext::new();
        assert!(!context.is_authenticated());
        assert!(!context.has_valid_session());
        assert_eq!(context.risk_score, 0.0);
    }

    #[tokio::test]
    async fn test_security_decision_checks() {
        assert!(SecurityDecision::Allow.is_allowed());
        assert!(!SecurityDecision::Allow.is_denied());

        let deny = SecurityDecision::Deny { reason: "test".to_string() };
        assert!(!deny.is_allowed());
        assert!(deny.is_denied());

        let challenge = SecurityDecision::Challenge { challenge_type: "captcha".to_string() };
        assert!(challenge.requires_challenge());

        let throttle = SecurityDecision::Throttle { delay_ms: 1000 };
        assert!(throttle.requires_throttle());
    }

    #[tokio::test]
    async fn test_security_service_builder() {
        let service = SecurityService::builder()
            .with_config(SecurityServiceConfig {
                enable_auth: true,
                enable_rate_limiting: true,
                security_level: 5,
                ..Default::default()
            })
            .build()
            .await
            .unwrap();

        assert_eq!(service.config.security_level, 5);
        assert!(service.config.enable_auth);
    }
}