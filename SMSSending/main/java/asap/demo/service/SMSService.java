package asap.demo.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.nurigo.sdk.NurigoApp;
import net.nurigo.sdk.message.model.Message;
import net.nurigo.sdk.message.request.SingleMessageSendingRequest;
import net.nurigo.sdk.message.response.SingleMessageSentResponse;
import net.nurigo.sdk.message.service.DefaultMessageService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;

@Slf4j
@Service
@RequiredArgsConstructor
public class SMSService {

    @Value("${coolsms.api-key}")
    private String apiKey;

    @Value("${coolsms.api-secret}")
    private String apiSecret;

    @Value("${coolsms.sender}")
    private String sender;

    private DefaultMessageService messageService;

    @PostConstruct
    public void init() {
        // 반드시 계정 내 등록된 유효한 API Key, API Secret Key를 입력해주셔야 합니다!
        this.messageService = NurigoApp.INSTANCE.initialize(apiKey, apiSecret, "https://api.coolsms.co.kr");
        log.info("CoolSMS messageService initialized with sender: {}", sender);
    }

    public void sendSMS(String receiver, String text) {
        try {
            log.info("Attempting to send SMS via CoolSMS to: {}", receiver);
            Message message = new Message();
            message.setFrom(sender); // 발신번호
            message.setTo(receiver); // 수신번호
            message.setText(text);   // 문자 내용

            SingleMessageSentResponse response = this.messageService.sendOne(new SingleMessageSendingRequest(message));
            log.info("CoolSMS 전송 응답: {}", response);

            if (response != null && response.getStatusCode() != null && response.getStatusCode().startsWith("2")) {
                log.info("SMS sent successfully to: {}. Message ID: {}", receiver, response.getMessageId());
            } else {
                log.error("Failed to send SMS to: {}. Response: {}", receiver, response);
                throw new RuntimeException("Failed to send SMS. Check logs for response details.");
            }
        } catch (Exception e) {
            log.error("Failed to send SMS to: {}", receiver, e);
            throw new RuntimeException("Failed to send SMS", e);
        }
    }
}