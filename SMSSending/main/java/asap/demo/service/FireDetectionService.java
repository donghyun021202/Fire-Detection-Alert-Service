package asap.demo.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Service
@RequiredArgsConstructor
public class FireDetectionService {
    private final SMSService smsService;

    public void handleFireDetection() {
        log.info("화재 감지 이벤트 처리 시작");
        
        log.info("화재 감지 이벤트가 발생했습니다. 감지 시간: {}",
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        
        log.info("화재 감지 이벤트 처리 완료");
    }
} 