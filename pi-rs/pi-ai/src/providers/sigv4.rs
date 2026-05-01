//! AWS SigV4 request signing utilities.
//!
//! Implements AWS Signature Version 4 signing for requests to AWS services like Bedrock.
//! Based on: https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html

use anyhow::Result;
use sha2::{Sha256, Digest};
use std::time::{SystemTime, UNIX_EPOCH};
use hmac::{Hmac, Mac};

type HmacSha256 = Hmac<Sha256>;

/// AWS SigV4 signing parameters
#[derive(Debug, Clone)]
pub struct SigV4Params {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub region: String,
    pub service: String,
}

/// Signed request with headers and signature
#[derive(Debug, Clone)]
pub struct SignedRequest {
    pub authorization_header: String,
    pub x_amz_date: String,
    pub x_amz_security_token: Option<String>,
}

/// Sign an AWS request using SigV4
pub fn sign_request(
    method: &str,
    url: &str,
    headers: &[(String, String)],
    body: &str,
    params: &SigV4Params,
) -> Result<SignedRequest> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs();
    
    let amz_date = format_timestamp(now);
    let datestamp = amz_date[..8].to_string();
    
    // Parse URL
    let url_obj = url::Url::parse(url)?;
    let host = url_obj.host_str().ok_or_else(|| anyhow::anyhow!("Invalid host in URL"))?;
    let path = url_obj.path();
    let query = url_obj.query().unwrap_or("");
    
    // 1. Create canonical request
    let canonical_request = create_canonical_request(
        method,
        path,
        query,
        &amz_date,
        host,
        headers,
        body,
        &params.session_token,
    );
    
    // 2. Create string to sign
    let credential_scope = format!("{}/{}/{}/aws4_request", 
        &datestamp, &params.region, &params.service);
    let string_to_sign = create_string_to_sign(&canonical_request, &amz_date, &credential_scope);
    
    // 3. Calculate signature
    let signature = calculate_signature(
        &string_to_sign,
        &params.secret_key,
        &datestamp,
        &params.region,
        &params.service,
    )?;
    
    // 4. Build Authorization header
    let signed_headers = build_signed_headers_list(headers, &params.session_token);
    let authorization_header = format!(
        "AWS4-HMAC-SHA256 Credential={}/aws4_request, SignedHeaders={}, Signature={}",
        format_credential(&params.access_key, &credential_scope),
        signed_headers,
        signature
    );
    
    Ok(SignedRequest {
        authorization_header,
        x_amz_date: amz_date,
        x_amz_security_token: params.session_token.clone(),
    })
}

/// Format timestamp as AWS expects (YYYYMMDDTHHMMSSZ)
fn format_timestamp(secs: u64) -> String {
    let secs_in_day = 86400;
    let days_since_epoch = secs / secs_in_day;
    let seconds_today = secs % secs_in_day;
    
    let hours = seconds_today / 3600;
    let minutes = (seconds_today % 3600) / 60;
    let seconds = seconds_today % 60;
    
    // Simple epoch to date calculation (works for 1970-2099)
    let total_days = days_since_epoch;
    let mut year = 1970;
    let mut days_left = total_days;
    
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days_left < days_in_year {
            break;
        }
        days_left -= days_in_year;
        year += 1;
    }
    
    let mut month = 1;
    let mut day = days_left + 1;
    let days_in_month_arr = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    
    for (i, &days_in_m) in days_in_month_arr.iter().enumerate() {
        if day <= days_in_m {
            month = i + 1;
            break;
        }
        day -= days_in_m;
    }
    
    format!(
        "{:04}{:02}{:02}T{:02}{:02}{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Create canonical request for signing
fn create_canonical_request(
    method: &str,
    path: &str,
    query: &str,
    amz_date: &str,
    host: &str,
    headers: &[(String, String)],
    body: &str,
    session_token: &Option<String>,
) -> String {
    let body_hash = sha256_hash(body);
    
    // Canonical headers
    let mut canonical_headers = String::new();
    canonical_headers.push_str(&format!("host:{}\n", host));
    canonical_headers.push_str(&format!("x-amz-date:{}\n", amz_date));
    
    if session_token.is_some() {
        canonical_headers.push_str(&format!("x-amz-security-token:{}\n", session_token.as_ref().unwrap()));
    }
    
    // Additional headers
    let mut sorted_headers: Vec<_> = headers.iter().collect();
    sorted_headers.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
    
    for (key, value) in sorted_headers {
        let key_lower = key.to_lowercase();
        if key_lower != "authorization" && key_lower != "host" && key_lower != "x-amz-date" && key_lower != "x-amz-security-token" {
            canonical_headers.push_str(&format!("{}:{}\n", key_lower, value.trim()));
        }
    }
    
    // Signed headers list
    let signed_headers_list = build_signed_headers_list(headers, session_token);
    
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method,
        path,
        query,
        canonical_headers,
        signed_headers_list,
        body_hash
    )
}

/// Build list of signed headers
fn build_signed_headers_list(headers: &[(String, String)], session_token: &Option<String>) -> String {
    let mut signed_headers = vec![
        "host".to_string(),
        "x-amz-date".to_string(),
    ];
    
    if session_token.is_some() {
        signed_headers.push("x-amz-security-token".to_string());
    }
    
    for (key, _) in headers {
        let key_lower = key.to_lowercase();
        if key_lower != "authorization" && key_lower != "host" && key_lower != "x-amz-date" && key_lower != "x-amz-security-token" {
            signed_headers.push(key_lower);
        }
    }
    
    signed_headers.sort();
    signed_headers.join(";")
}

/// SHA256 hash of a string
fn sha256_hash(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Create string to sign
fn create_string_to_sign(canonical_request: &str, amz_date: &str, credential_scope: &str) -> String {
    let canonical_request_hash = sha256_hash(canonical_request);
    format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date, credential_scope, canonical_request_hash
    )
}

/// Calculate SigV4 signature
fn calculate_signature(
    string_to_sign: &str,
    secret_key: &str,
    datestamp: &str,
    region: &str,
    service: &str,
) -> Result<String> {
    let k_date = sign(&format!("AWS4{}", secret_key), datestamp)?;
    let k_region = sign(&k_date, region)?;
    let k_service = sign(&k_region, service)?;
    let k_signing = sign(&k_service, "aws4_request")?;
    sign(&k_signing, string_to_sign)
}

/// HMAC-SHA256 signing
fn sign(key: &str, msg: &str) -> Result<String> {
    let mut mac = HmacSha256::new_from_slice(key.as_bytes())
        .map_err(|e| anyhow::anyhow!("Failed to create HMAC: {}", e))?;
    mac.update(msg.as_bytes());
    let result = mac.finalize();
    Ok(result.into_bytes().iter().map(|b| format!("{:02x}", b)).collect::<String>())
}

fn format_credential(access_key: &str, credential_scope: &str) -> String {
    format!("{}/{}", access_key, credential_scope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2004));
        assert!(!is_leap_year(1900));
        assert!(!is_leap_year(2001));
    }

    #[test]
    fn test_sha256_hash() {
        let hash = sha256_hash("");
        assert_eq!(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_sign_request() {
        let params = SigV4Params {
            access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
            region: "us-east-1".to_string(),
            service: "bedrock-runtime".to_string(),
        };

        let headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let result = sign_request(
            "POST",
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
            &headers,
            "{}",
            &params,
        );

        assert!(result.is_ok());
        let signed = result.unwrap();
        assert!(signed.authorization_header.contains("AWS4-HMAC-SHA256"));
        assert!(signed.authorization_header.contains("Credential="));
        assert!(!signed.x_amz_date.is_empty());
    }
}
