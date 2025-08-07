package asap.demo.controller;

import asap.demo.service.FireDetectionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/api/fire-detection")
@RequiredArgsConstructor
public class FireDetectionController {
    private final FireDetectionService fireDetectionService;

    @PostMapping("/detect")
    public ResponseEntity<String> detectFire(@RequestParam boolean isFire) {
        log.info("화재 감지 요청 수신 - 화재 발생: {}", isFire);
        
        if (!isFire) {
            return ResponseEntity.ok("화재가 감지되지 않았습니다.");
        }
        
        try {
            fireDetectionService.handleFireDetection();
            return ResponseEntity.ok("화재 감지 알림이 발송되었습니다.");
        } catch (Exception e) {
            log.error("화재 감지 처리 중 오류 발생", e);
            return ResponseEntity.internalServerError().body("화재 감지 알림 발송에 실패했습니다.");
        }
    }
} 